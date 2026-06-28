# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@description: LeadRegionViewNet + rhythm-frequency view + morphology-gradient view = MORSE-Net (8视图)
    
核心设计：
    - 保留6个导联区域视图分支（多导联分组）
    - 第7个视图：时频融合分支（全12导联）
    - 第8个视图：Sobel形态梯度分支（全12导联一阶微分）← 新增创新点
    - 统一融合机制：8个视图的自适应权重融合
    
架构图：
    12导联 ECG 输入
    ├─ 视图1-6: 导联分组（TemporalRes2NetBackbone 分支） → 128维特征
    ├─ 视图7: 全导联时频 (LearnableSTFT + SpectrogramCNN) → 128维特征
    └─ 视图8: Sobel形态梯度 (Sobel Conv → TemporalRes2NetBackbone) → 128维特征 ← 新增
         ↓
    8个视图的自适应权重
         ↓
    加权融合 → 128维融合特征
         ↓
    分类头 → num_classes 输出
    
Sobel形态梯度视图创新点：
    心电信号的多标签诊断（束支阻滞、ST段异常等）高度依赖波形的斜率变化。
    Sobel算子本质上是高效的一阶微分，通过不可学习的 [-1, 0, 1] 卷积核对每个
    导联独立提取梯度信号，再经TemporalRes2NetBackbone 骨干网络提取深层特征。
    这是传统信号处理先验（微分算子）与深度学习的有机结合，
    强制模型从"变化率"角度审视心电图，而非仅依赖原始幅值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_layers import CoordAtt, CrossModalAttention
from models.temporal_backbone import Mish, Res2Block, AdaptiveViewWeight, TemporalRes2NetBackbone
from models.rhythm_frequency_branch import LearnableSTFT, SpectrogramCNN
from models.diagnostic_label_graph import LabelGraphRefiner, FeatureLabelGCN


def _remap_state_dict_keys(state_dict, prefix_pairs):
    """Return a state_dict with legacy module prefixes rewritten to current names."""
    from collections import OrderedDict

    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_pairs:
            if key.startswith(old_prefix + '.'):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value

    if hasattr(state_dict, '_metadata'):
        remapped._metadata = state_dict._metadata
    return remapped


class RhythmFrequencyView(nn.Module):
    """时频融合视图分支：输入12导联→输出128维特征
    
    设计思路：
        1. LearnableSTFT：多尺度频谱分析
        2. SpectrogramCNN：频谱特征提取
        3. 特征压缩：降到128维，匹配其他视图输出
    """
    
    def __init__(self, input_channels=12, output_dim=128,
                 use_lead_attention=True, lead_attention_reduction=4,
                 lead_attention_spectral_spatial=True, lead_attention_spatial_kernel=7):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # 多尺度STFT（输入12导联）
        self.stft = LearnableSTFT(
            n_fft=512,
            hop_length=128,
            n_scales=3
        )
        
        # 谱图CNN分支（输入通道=12，输出特征=512）
        self.spectrogram_cnn = SpectrogramCNN(
            input_channels=input_channels,
            num_classes=output_dim,  # 直接输出目标维度，避免额外映射
            hidden_dims=[64, 128, 256, 512],
            use_lead_attention=use_lead_attention,
            lead_attention_reduction=lead_attention_reduction,
            lead_attention_spectral_spatial=lead_attention_spectral_spatial,
            lead_attention_spatial_kernel=lead_attention_spatial_kernel,
        )
        
        # 特征投影层（512→128维）
        self.feature_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
        
        # 全局特征处理（确保稳定性）
        self.output_norm = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        """
        参数：
            x: 输入信号，形状 (B, 12, L)
        返回：
            feat: 时频特征，形状 (B, 128)
        """
        # 获取多尺度谱图
        rhythm_spectrogram, _ = self.stft(x)  # (B, 12, n_freqs, T)
        
        # 谱图CNN提取特征
        rhythm_spectral_feature, _, lead_weights = self.spectrogram_cnn(rhythm_spectrogram)  # (B, 512)
        
        # 特征投影到128维
        feat = self.feature_projection(rhythm_spectral_feature)  # (B, 128)
        
        # 归一化输出
        feat = self.output_norm(feat)
        
        return feat


class MorphologyGradientView(nn.Module):
    """Sobel引导的形态梯度视图：通过一阶微分算子提取ECG波形斜率特征
    
    核心思想：
        心电信号的多标签诊断（束支阻滞、ST段异常等）高度依赖波形的斜率变化。
        Sobel算子本质上是高效的一阶微分，可以强制模型从"变化率"视角审视心电图，
        而非仅依赖原始幅值。
    
    实现细节：
        1. 不可学习的1D Sobel卷积 [-1, 0, 1]，对每个导联独立提取一阶梯度
        2. 梯度信号通过 TemporalRes2NetBackbone 骨干网络提取128维深层特征表示
    
    创新性：
        传统信号处理先验（微分算子）与深度学习的有机结合，
        为模型提供显式的"形态变化率"视角，与幅值域视图形成互补。
    """
    
    def __init__(self, input_channels=12, output_dim=128, num_classes=9):
        super().__init__()
        self.input_channels = input_channels
        
        # ===== 不可学习的1D Sobel卷积 =====
        # 使用 F.conv1d + register_buffer，而非 nn.Conv1d + nn.Parameter
        # 这样 Sobel 核不会出现在 model.parameters() 中，不会被优化器跟踪
        sobel_kernel = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 3)
        sobel_kernel = sobel_kernel.repeat(input_channels, 1, 1)  # (12, 1, 3)
        self.register_buffer('sobel_kernel', sobel_kernel)  # 自动跟随 .to(device)
        
        # ===== 【关键修复】梯度信号归一化 =====
        # Sobel输出的一阶差分信号统计分布与原始ECG幅值信号差异巨大：
        #   - 原始信号：均值非零，方差取决于ECG幅度
        #   - 梯度信号：近零均值，方差极小（平坦段）或突变（QRS波群）
        # 不归一化会导致骨干网络第一层conv1(kernel=25)收到分布错误的输入，
        # 使得整个Sobel分支学出噪声特征，反向污染融合结果。
        # BatchNorm1d 逐通道（逐导联）归一化梯度信号到标准分布。
        self.gradient_norm = nn.BatchNorm1d(input_channels)
        
        # ===== 特征提取骨干：复用 TemporalRes2NetBackbone 架构处理梯度信号 =====
        # 输入为12通道归一化梯度信号，输出128维特征
        self.backbone = TemporalRes2NetBackbone(
            input_channels=input_channels,
            single_view=True,
            num_classes=num_classes
        )
    
    def forward(self, x, return_seq=False):
        """前向传播
        
        参数：
            x: 原始12导联ECG信号，形状 (B, 12, L)
            return_seq: 是否返回池化前的序列特征（用于跨模态融合）
        返回：
            feat: 形态梯度特征，形状 (B, 128)
            seq_feat: （可选）池化前序列特征，形状 (B, 128, L'')
        """
        # 第一步：Sobel一阶微分 → 提取各导联的斜率/梯度特征图
        # 使用 F.conv1d + buffer（不经过 nn.Conv1d，避免权重被优化器跟踪）
        gradient = F.conv1d(
            x, self.sobel_kernel, bias=None,
            stride=1, padding=1, groups=self.input_channels
        )  # (B, 12, L)
        
        # 第二步：【关键】归一化梯度信号，使其分布适配骨干网络
        gradient = self.gradient_norm(gradient)  # (B, 12, L)
        
        # 第三步：通过骨干网络提取深层特征
        return self.backbone(gradient, return_seq=return_seq)


class MORSENet(nn.Module):
    """多视图ECG分类模型：6个导联视图 + 1个时频融合视图 + 1个Sobel形态梯度视图
    
    视图划分：
        视图1: 1导联（I）
        视图2: 2导联（aVR, aVL）
        视图3: 2导联（V1, V2）
        视图4: 2导联（V3, V4）
        视图5: 2导联（V5, V6）
        视图6: 3导联（II, III, aVF）
        视图7: 12导联（全导联时频融合）
        视图8: 12导联（Sobel形态梯度） ← 新增
    
    融合策略：
        - 每个视图生成自适应权重
        - 加权融合8个视图特征
        - 最终分类
    """
    
    def __init__(
            self,
            num_classes=9,
            use_label_graph_refiner=False,
            label_graph_hidden=64,
            label_graph_learnable_adj=True,
            label_graph_dropout=0.1,
            use_feature_label_gcn=True,
            feature_label_gcn_hidden=64,
            feature_label_gcn_layers=2,
            feature_label_gcn_dropout=0.1,
            feature_label_gcn_learnable_adj=True,
            feature_label_gcn_init_gate=-2.0,
            feature_label_gcn_adj_init_off_diag=0.1,
            use_view_transformer_fusion=True,
            view_transformer_layers=1,
            view_transformer_heads=4,
            view_transformer_dropout=0.1,
            view_transformer_residual_scale=0.1,
            use_cross_modal_fusion=True,
            cross_modal_heads=4,
            cross_modal_dropout=0.1,
            cross_modal_tokens=32,
            use_lead_attention=True,
            lead_attention_reduction=4,
            lead_attention_spectral_spatial=True,
            lead_attention_spatial_kernel=7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_view_transformer_fusion = use_view_transformer_fusion
        self.view_transformer_residual_scale = view_transformer_residual_scale
        self.use_cross_modal_fusion = use_cross_modal_fusion
        
        # ========== 6个导联区域视图分支 ==========
        # 视图1-6的TemporalRes2NetBackbone 分支（保持原有设计，single_view=True输出特征）
        self.lead_i_branch = TemporalRes2NetBackbone(input_channels=1, single_view=True, num_classes=num_classes)   # 1导联
        self.augmented_limb_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.septal_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.anterior_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.lateral_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True, num_classes=num_classes)   # 2导联
        self.inferior_limb_branch = TemporalRes2NetBackbone(input_channels=3, single_view=True, num_classes=num_classes)   # 3导联
        
        # ========== 新增：第7个视图 - 时频融合 ==========
        self.rhythm_frequency_view = RhythmFrequencyView(
            input_channels=12, output_dim=128,
            use_lead_attention=use_lead_attention,
            lead_attention_reduction=lead_attention_reduction,
            lead_attention_spectral_spatial=lead_attention_spectral_spatial,
            lead_attention_spatial_kernel=lead_attention_spatial_kernel,
        )
        
        # ========== 新增：第8个视图 - Sobel形态梯度 ==========
        self.morphology_gradient_view = MorphologyGradientView(
            input_channels=12, output_dim=128, num_classes=num_classes
        )
        # 【关键】Sobel视图渐进式门控：初始化为极小贡献
        # sigmoid(-3) ≈ 0.047，意味着训练初期Sobel视图仅贡献~5%的融合权重，
        # 随着骨干网络学到有用的梯度特征，门控自动打开，
        # 避免训练初期噪声特征污染其他7个已知视图的融合。
        self.sobel_gate = nn.Parameter(torch.tensor(-3.0))
        
        # ========== 8个视图的自适应权重模块 ==========
        self.lead_i_weight = AdaptiveViewWeight(128)
        self.augmented_limb_weight = AdaptiveViewWeight(128)
        self.septal_precordial_weight = AdaptiveViewWeight(128)
        self.anterior_precordial_weight = AdaptiveViewWeight(128)
        self.lateral_precordial_weight = AdaptiveViewWeight(128)
        self.inferior_limb_weight = AdaptiveViewWeight(128)
        self.rhythm_frequency_weight = AdaptiveViewWeight(128)  # 时频视图的权重
        self.morphology_gradient_weight = AdaptiveViewWeight(128)  # 新增：Sobel形态梯度视图的权重

        # ========== 新增：视图级 Transformer 融合（8个视图token） ==========
        self.view_pos_embed = None
        self.view_transformer = None
        self.view_transformer_norm = None
        if self.use_view_transformer_fusion:
            self.view_pos_embed = nn.Parameter(torch.zeros(1, 8, 128))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=view_transformer_heads,
                dim_feedforward=256,
                dropout=view_transformer_dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True,
            )
            self.view_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=view_transformer_layers,
            )
            self.view_transformer_norm = nn.LayerNorm(128)

        # ========== 跨模态中融合 (Cross-Modal Mid-Fusion) ==========
        # 在网络中层引入时域-频域双向交叉注意力
        self.cross_modal_attn = None
        self.freq_seq_encoder = None
        self.time_seq_downsample = None
        self.cross_modal_gate = None
        if self.use_cross_modal_fusion:
            # 双向交叉注意力模块
            self.cross_modal_attn = CrossModalAttention(
                d_model=128,
                nhead=cross_modal_heads,
                dropout=cross_modal_dropout,
            )
            # 频域序列编码器：从 STFT 谱图中提取频域 token 序列
            self.freq_seq_encoder = nn.Sequential(
                nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # 时域序列降采样：每个视图的序列压缩到固定 token 数，控制 cross-attention 计算开销
            self.time_seq_downsample = nn.AdaptiveAvgPool1d(cross_modal_tokens)
            # 可学习的融合门控（初始化为较小值，训练初期让原始特征主导）
            self.cross_modal_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12

        # ========== Feature-level 标签图 GCN（新增，作用于 fused_feat 上） ==========
        self.feature_label_gcn = None
        if use_feature_label_gcn:
            self.feature_label_gcn = FeatureLabelGCN(
                feat_dim=128,
                num_classes=num_classes,
                gcn_hidden=feature_label_gcn_hidden,
                num_gcn_layers=feature_label_gcn_layers,
                dropout=feature_label_gcn_dropout,
                learnable_adj=feature_label_gcn_learnable_adj,
                init_gate=feature_label_gcn_init_gate,
                adj_init_off_diag=feature_label_gcn_adj_init_off_diag,
            )

        # 最终分类头
        self.fc = nn.Linear(128, num_classes)

        # Logits-level 标签图细化模块（可选，原有机制保留）
        self.label_refiner = None
        if use_label_graph_refiner:
            self.label_refiner = LabelGraphRefiner(
                num_classes=num_classes,
                hidden=label_graph_hidden,
                dropout=label_graph_dropout,
                learnable_adj=label_graph_learnable_adj,
            )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_intermediate=False):
        """前向传播：8视图特征提取→自适应加权融合→分类
        
        参数：
            x: 输入12导联ECG信号，形状 (batch_size, 12, seq_len)
            return_intermediate: 是否返回中间特征（用于可视化）
        
        返回：
            logits: 分类输出，形状 (batch_size, num_classes)
            intermediate: （可选）中间特征字典
        """
        
        # ========== 准备各视图的输入 ==========
        lead_i_signal = x[:, 3, :].unsqueeze(1)                                               # 导联3 (I)
        augmented_limb_signal = torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)  # 导联0,4 (aVR, aVL)
        septal_precordial_signal = x[:, 6:8, :]                                                           # 导联6-7 (V1, V2)
        anterior_precordial_signal = x[:, 8:10, :]                                                          # 导联8-9 (V3, V4)
        lateral_precordial_signal = x[:, 10:12, :]                                                         # 导联10-11 (V5, V6)
        inferior_limb_signal = torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1)             # 导联1-2,5 (II, III, aVF)

        # ========== 第一步：特征提取 + 跨模态中融合 (Mid-Fusion) ==========
        if self.use_cross_modal_fusion and self.cross_modal_attn is not None:
            # --- 时域视图：同时获取池化后特征和池化前序列特征 ---
            lead_i_feature, lead_i_sequence = self.lead_i_branch(lead_i_signal, return_seq=True)
            augmented_limb_feature, augmented_limb_sequence = self.augmented_limb_branch(augmented_limb_signal, return_seq=True)
            septal_precordial_feature, septal_precordial_sequence = self.septal_precordial_branch(septal_precordial_signal, return_seq=True)
            anterior_precordial_feature, anterior_precordial_sequence = self.anterior_precordial_branch(anterior_precordial_signal, return_seq=True)
            lateral_precordial_feature, lateral_precordial_sequence = self.lateral_precordial_branch(lateral_precordial_signal, return_seq=True)
            inferior_limb_feature, inferior_limb_sequence = self.inferior_limb_branch(inferior_limb_signal, return_seq=True)

            # --- Sobel形态梯度视图：全导联一阶微分特征 ---
            morphology_gradient_feature, morphology_gradient_sequence = self.morphology_gradient_view(x, return_seq=True)

            # --- 频域视图：计算 STFT 一次并复用（避免重复计算） ---
            rhythm_spectrogram, _ = self.rhythm_frequency_view.stft(x)
            rhythm_spectral_feature, _, _ = self.rhythm_frequency_view.spectrogram_cnn(rhythm_spectrogram)
            rhythm_frequency_feature = self.rhythm_frequency_view.feature_projection(rhythm_spectral_feature)
            rhythm_frequency_feature = self.rhythm_frequency_view.output_norm(rhythm_frequency_feature)

            # --- 跨模态交叉注意力 (Cross-Modal Mid-Fusion) ---
            # 时域: 降采样各视图序列（含Sobel视图），拼接为时域 token 序列
            temporal_sequences = [lead_i_sequence, augmented_limb_sequence, septal_precordial_sequence,
                         anterior_precordial_sequence, lateral_precordial_sequence, inferior_limb_sequence, morphology_gradient_sequence]
            temporal_sequences_ds = [self.time_seq_downsample(seq) for seq in temporal_sequences]
            T_ds = temporal_sequences_ds[0].shape[2]  # 每个视图降采样后的 token 数
            temporal_token_map = torch.cat(temporal_sequences_ds, dim=2)         # (B, 128, 7*T_ds)
            time_tokens = temporal_token_map.permute(0, 2, 1)           # (B, 7*T_ds, 128)

            # 频域: 从缓存的 STFT 谱图编码频域 token 序列
            freq_map = self.freq_seq_encoder(rhythm_spectrogram)          # (B, 128, F', T')
            freq_tokens = freq_map.flatten(2).permute(0, 2, 1)   # (B, S, 128)

            # 双向交叉注意力：
            #   频域Q → 时域K/V: 节律异常 → 寻找对应异常波形
            #   时域Q → 频域K/V: 波形形态 → 寻找对应频率成分
            time_enhanced, freq_enhanced = self.cross_modal_attn(time_tokens, freq_tokens)

            # 将增强后的 token 池化为修正向量，通过可学习门控注入原始特征
            gate = torch.sigmoid(self.cross_modal_gate)

            # 时域修正: 拆回7个时域视图（6个导联视图+1个Sobel视图），分别池化为修正向量
            time_enh_split = time_enhanced.permute(0, 2, 1)       # (B, 128, 7*T_ds)
            time_view_chunks = torch.split(time_enh_split, T_ds, dim=2)
            lead_i_feature = lead_i_feature + gate * time_view_chunks[0].mean(dim=2)
            augmented_limb_feature = augmented_limb_feature + gate * time_view_chunks[1].mean(dim=2)
            septal_precordial_feature = septal_precordial_feature + gate * time_view_chunks[2].mean(dim=2)
            anterior_precordial_feature = anterior_precordial_feature + gate * time_view_chunks[3].mean(dim=2)
            lateral_precordial_feature = lateral_precordial_feature + gate * time_view_chunks[4].mean(dim=2)
            inferior_limb_feature = inferior_limb_feature + gate * time_view_chunks[5].mean(dim=2)
            morphology_gradient_feature = morphology_gradient_feature + gate * time_view_chunks[6].mean(dim=2)  # Sobel时域修正

            # 频域修正
            rhythm_frequency_feature = rhythm_frequency_feature + gate * freq_enhanced.permute(0, 2, 1).mean(dim=2)
        else:
            # --- 原始路径（无跨模态中融合，纯后融合） ---
            lead_i_feature = self.lead_i_branch(lead_i_signal)
            augmented_limb_feature = self.augmented_limb_branch(augmented_limb_signal)
            septal_precordial_feature = self.septal_precordial_branch(septal_precordial_signal)
            anterior_precordial_feature = self.anterior_precordial_branch(anterior_precordial_signal)
            lateral_precordial_feature = self.lateral_precordial_branch(lateral_precordial_signal)
            inferior_limb_feature = self.inferior_limb_branch(inferior_limb_signal)
            rhythm_frequency_feature = self.rhythm_frequency_view(x)
            morphology_gradient_feature = self.morphology_gradient_view(x)  # Sobel形态梯度视图

        # ========== 第二步：计算8个视图的自适应权重 ==========
        representation_features = [lead_i_feature, augmented_limb_feature, septal_precordial_feature, anterior_precordial_feature,
                         lateral_precordial_feature, inferior_limb_feature, rhythm_frequency_feature, morphology_gradient_feature]

        lead_i_score = self.lead_i_weight(lead_i_feature)  # (batch_size, 1)
        augmented_limb_score = self.augmented_limb_weight(augmented_limb_feature)
        septal_precordial_score = self.septal_precordial_weight(septal_precordial_feature)
        anterior_precordial_score = self.anterior_precordial_weight(anterior_precordial_feature)
        lateral_precordial_score = self.lateral_precordial_weight(lateral_precordial_feature)
        inferior_limb_score = self.inferior_limb_weight(inferior_limb_feature)
        rhythm_frequency_score = self.rhythm_frequency_weight(rhythm_frequency_feature)  # 时频视图权重
        morphology_gradient_score = self.morphology_gradient_weight(morphology_gradient_feature)  # Sobel形态梯度视图权重
        
        representation_weights = [
            lead_i_score, augmented_limb_score, septal_precordial_score, anterior_precordial_score,
            lateral_precordial_score, inferior_limb_score, rhythm_frequency_score, morphology_gradient_score
        ]

        # ========== 第三步：融合8个视图 ==========
        # Sobel渐进式门控：控制第8视图对融合的贡献比例
        sobel_gate = torch.sigmoid(self.sobel_gate)  # 初始≈ 0.047, 渐进增大
        
        # 先计算原始加权融合（作为主干融合特征）
        fused_sum = (lead_i_score * lead_i_feature +
                     augmented_limb_score * augmented_limb_feature +
                     septal_precordial_score * septal_precordial_feature +
                     anterior_precordial_score * anterior_precordial_feature +
                     lateral_precordial_score * lateral_precordial_feature +
                     inferior_limb_score * inferior_limb_feature +
                     rhythm_frequency_score * rhythm_frequency_feature +
                     sobel_gate * morphology_gradient_score * morphology_gradient_feature)  # (B, 128)  门控叠加在Sobel视图上

        if self.use_view_transformer_fusion and self.view_transformer is not None:
            # (B, 8, 128) 视图token，Sobel视图经门控缩放
            representation_features_gated = [
                lead_i_feature, augmented_limb_feature, septal_precordial_feature, anterior_precordial_feature,
                lateral_precordial_feature, inferior_limb_feature, rhythm_frequency_feature,
                sobel_gate * morphology_gradient_feature  # 门控也作用于Transformer token
            ]
            representation_tokens = torch.stack(representation_features_gated, dim=1)

            # 使用softmax权重做"温和缩放"，避免token幅值差异过大导致不稳定
            weights_raw = torch.cat(representation_weights, dim=1)  # (B, 8)
            weights = F.softmax(weights_raw, dim=1)  # (B, 8)
            token_scale = 0.5 + 0.5 * weights  # (B, 8) in [0.5, 1.0]
            representation_tokens = representation_tokens * token_scale.unsqueeze(-1)

            # 视图位置编码
            if self.view_pos_embed is not None:
                representation_tokens = representation_tokens + self.view_pos_embed

            # Transformer 编码得到修正量（delta），再用残差方式叠加回 fused_sum
            representation_tokens = self.view_transformer(representation_tokens)
            delta = representation_tokens.mean(dim=1)  # (B, 128)
            delta = self.view_transformer_norm(delta)
            fused_feat = fused_sum + self.view_transformer_residual_scale * delta
        else:
            fused_feat = fused_sum
        
        # ========== 第四步：Feature-level 标签图建模 + 最终分类 ==========
        # 先在特征空间做标签共现图卷积（如果启用）
        if self.feature_label_gcn is not None:
            fused_feat = self.feature_label_gcn(fused_feat)  # (B, 128) → 标签图增强 → (B, 128)

        logits = self.fc(fused_feat)  # (batch_size, num_classes)

        # Logits-level 后处理（如果同时启用）
        if self.label_refiner is not None:
            logits = self.label_refiner(logits)
        
        # ========== 返回结果 ==========
        if return_intermediate:
            intermediate = {
                'representation_features': representation_features,
                'representation_weights': representation_weights,
                # Backward-compatible keys consumed by distillation/plot scripts.
                'view_features': representation_features,
                'fuse_weights': representation_weights,
                'fused_feat': fused_feat,
                'view_names': ['lead_I', 'augmented_limb_aVR_aVL', 'septal_V1_V2', 
                              'anterior_V3_V4', 'lateral_V5_V6', 'inferior_II_III_aVF', 
                              'rhythm_frequency', 'morphology_gradient'],
                'cross_modal_gate': torch.sigmoid(self.cross_modal_gate).item() if self.use_cross_modal_fusion and self.cross_modal_gate is not None else None,
                'sobel_gate': sobel_gate.item(),
            }
            return logits, intermediate
        
        return logits
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        legacy_prefix_pairs = [
            ('MyNet1', 'lead_i_branch'),
            ('MyNet2', 'augmented_limb_branch'),
            ('MyNet3', 'septal_precordial_branch'),
            ('MyNet4', 'anterior_precordial_branch'),
            ('MyNet5', 'lateral_precordial_branch'),
            ('MyNet6', 'inferior_limb_branch'),
            ('MyNet7_TimeFreq', 'rhythm_frequency_view'),
            ('MyNet8_Sobel', 'morphology_gradient_view'),
            ('fuse_weight_1', 'lead_i_weight'),
            ('fuse_weight_2', 'augmented_limb_weight'),
            ('fuse_weight_3', 'septal_precordial_weight'),
            ('fuse_weight_4', 'anterior_precordial_weight'),
            ('fuse_weight_5', 'lateral_precordial_weight'),
            ('fuse_weight_6', 'inferior_limb_weight'),
            ('fuse_weight_7', 'rhythm_frequency_weight'),
            ('fuse_weight_8', 'morphology_gradient_weight'),
        ]
        state_dict = _remap_state_dict_keys(state_dict, legacy_prefix_pairs)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def get_view_weights(self, x):
        """获取8个视图的融合权重分布（用于可视化）
        
        返回：
            weights_np: numpy数组，形状 (batch_size, 8)，每行sum=1
        """
        _, intermediate = self.forward(x, return_intermediate=True)
        
        # 提取权重并进行softmax归一化
        weights = torch.cat(intermediate['representation_weights'], dim=1)  # (B, 8)
        weights = F.softmax(weights, dim=1)
        
        return weights.detach().cpu().numpy()


# ========== 便捷函数 ==========

def create_morse_net(num_classes=9, pretrained=False):
    """创建 MORSE-Net 模型
    
    参数：
        num_classes: 分类类别数
        pretrained: 是否加载预训练权重
    
    返回：
        model: MORSENet 实例
    """
    model = MORSENet(num_classes=num_classes)
    
    if pretrained:
        # TODO: 从checkpoints加载预训练权重
        pass
    
    return model


# ---------------------------------------------------------------------------
# Backward-compatible aliases.  Keep these for old configs/scripts while the
# refactored public API uses MORSE-Net naming.
TimeFreqView = RhythmFrequencyView
SobelMorphologicalView = MorphologyGradientView
MyNet7ViewTimeFreq = MORSENet
create_mynet7view_timefreq = create_morse_net
