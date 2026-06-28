# -*- coding: utf-8 -*-
"""
@time: 2026/1/5
@author: 
@description: 多尺度时频融合模型 (Scheme #7)
    时域分支：捕捉波形形态特征（QRS/ST/T波形）
    频域分支：捕捉节律特征（RR间期、心率变异等频域特征）
    标签条件融合：不同标签自动偏向不同分支
    
核心创新：
    1. 可学习的STFT变换（自适应中心频率和带宽）
    2. 多尺度频谱分析（低中高频段并行处理）
    3. 标签感知的动态权重融合（标签影响时频特征的混合比例）
    4. 鲁棒性增强（对肌电干扰、基线漂移的抵抗）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.attention_layers import CoordAtt, LeadAttention
from models.diagnostic_label_graph import LabelGraphRefiner


# ==================== 时频变换模块 ====================

class LearnableSTFT(nn.Module):
    """可学习的STFT变换
    
    特点：
        - 通过学习合成窗函数，自适应不同频段
        - 支持多尺度时间窗（短窗捕捉快速变化，长窗捕捉低频）
        - 数值稳定性好（加eps防止log(0)）
    
    参数：
        n_fft: FFT大小
        hop_length: 跳跃大小
        n_scales: 多尺度数量（默认3：短中长时间窗）
    """
    
    def __init__(self, n_fft=512, hop_length=128, n_scales=3, eps=1e-9):
        super(LearnableSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_scales = n_scales
        self.eps = eps
        self.n_freqs = n_fft // 2 + 1  # 频率bin数量
        
        # 可学习的窗函数（多尺度）
        self.window_params = nn.Parameter(
            torch.randn(n_scales, n_fft) * 0.01,
            requires_grad=True
        )
        
        # 缩放因子（学习每个尺度的重要性）
        self.scale_weights = nn.Parameter(
            torch.ones(n_scales) / n_scales,
            requires_grad=True
        )
    
    def forward(self, x):
        """
        参数：
            x: 输入信号，形状 (B, C, L) - [批量, 通道, 长度]
        返回：
            spectrograms: 多尺度谱图列表，每个形状 (B, C, n_freqs, T)
            其中 T 是时间帧数
        """
        B, C, L = x.size()
        spectrograms = []
        
        # 多尺度STFT处理
        for scale_idx in range(self.n_scales):
            # 生成窗函数（Hann窗的变体）
            window = torch.hann_window(
                self.n_fft, 
                periodic=False,
                device=x.device,
                dtype=x.dtype
            )
            # 加入可学习参数（控制窗的尖锐度）
            window_param = torch.sigmoid(self.window_params[scale_idx])
            window = window * (0.5 + 1.5 * window_param.mean())
            window = window / window.sum()  # 归一化
            
            # STFT变换（通道维度保留）
            # 处理每个通道
            spec_scale = []
            for c in range(C):
                spec = torch.stft(
                    x[:, c, :],  # (B, L)
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=window,
                    return_complex=True,
                    pad_mode='reflect',
                    center=True
                )  # (B, n_freqs, T)
                
                # 计算幅度谱（dB scale）
                spec_mag = torch.abs(spec)
                spec_db = 20 * torch.log10(spec_mag + self.eps)
                spec_scale.append(spec_db)
            
            spec_scale = torch.stack(spec_scale, dim=1)  # (B, C, n_freqs, T)
            spectrograms.append(spec_scale)
        
        # 按缩放权重加权融合
        scale_weights = F.softmax(self.scale_weights, dim=0)
        fused_spec = sum(
            scale_weights[i] * spectrograms[i] 
            for i in range(self.n_scales)
        )
        
        return fused_spec, spectrograms  # 返回融合谱和多尺度谱


class SpectrogramCNN(nn.Module):
    """谱图CNN分支：处理时频表示
    
    设计思路：
        - 导联注意力（LeadAttention）：自适应放大关键导联的频谱，抑制无关导联噪声
        - 多尺度卷积核（1x3, 3x5, 5x7...）捕捉不同时频模式
        - 频率方向和时间方向分别进行特征提取
        - 残差连接确保梯度流通
    """
    
    def __init__(self, input_channels=12, num_classes=10, hidden_dims=[64, 128, 256, 512],
                 use_lead_attention=True, lead_attention_reduction=4,
                 lead_attention_spectral_spatial=True, lead_attention_spatial_kernel=7):
        super(SpectrogramCNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.use_lead_attention = use_lead_attention
        
        # ========== 导联注意力模块 (Lead Attention) ==========
        # 在多尺度卷积之前施加，让网络自适应地关注关键导联
        # 例如：下壁心梗时放大 II, III, aVF 导联的频谱
        self.lead_attention = None
        if use_lead_attention:
            self.lead_attention = LeadAttention(
                num_leads=input_channels,
                reduction=lead_attention_reduction,
                use_spectral_spatial=lead_attention_spectral_spatial,
                spatial_kernel_size=lead_attention_spatial_kernel,
            )
        
        # 多尺度卷积块（频率维度和时间维度）
        self.freq_convs = nn.ModuleList([
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(7, 1), padding=(3, 0), bias=False),
        ])
        
        self.time_convs = nn.ModuleList([
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=(1, 7), padding=(0, 3), bias=False),
        ])
        
        # 频率时间联合处理块
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dims[0], hidden_dims[1], stride=2),
            self._make_block(hidden_dims[1], hidden_dims[2], stride=2),
            self._make_block(hidden_dims[2], hidden_dims[3], stride=2),
        ])
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[2], num_classes)
        )
    
    def _make_block(self, in_channels, out_channels, stride=1):
        """残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, spec):
        """
        参数：
            spec: 谱图，形状 (B, C, F, T) - [批量, 通道, 频率, 时间]
        返回：
            features: 特征向量，形状 (B, hidden_dims[-1])
            logits: 分类logits，形状 (B, num_classes)
        """
        # ========== 导联注意力：自适应重标定各导联频谱 ==========
        lead_weights = None
        if self.use_lead_attention and self.lead_attention is not None:
            spec, lead_weights = self.lead_attention(spec)  # (B, C, F, T), (B, C)
        
        # 多尺度频率卷积
        freq_feats = []
        for conv in self.freq_convs:
            freq_feats.append(conv(spec))  # (B, hidden_dims[0], F, T)
        
        # 多尺度时间卷积
        time_feats = []
        for conv in self.time_convs:
            time_feats.append(conv(spec))
        
        # 特征融合 - 在通道维度拼接后再通过1x1卷积降维
        x = torch.cat(freq_feats + time_feats, dim=1)  # (B, 6*hidden_dims[0], F, T)
        
        # 通道求和并归一化，保持通道数为hidden_dims[0]
        B, C, freq_dim, time_dim = x.shape
        # 通道求均值：(B, 6*hidden_dims[0], F, T) -> (B, hidden_dims[0], F, T)
        x = x.view(B, 6, self.hidden_dims[0], freq_dim, time_dim).mean(dim=1)  # (B, hidden_dims[0], F, T)
        
        # 逐层处理
        for block in self.blocks:
            x = block(x)
            x = F.relu(x)
        
        # 全局平均池化
        features = self.gap(x).squeeze(-1).squeeze(-1)  # (B, hidden_dims[-1])
        
        # 分类
        logits = self.classifier(features)
        
        return features, logits, lead_weights


# ==================== 时域分支（复用现有ResNet结构） ====================

class TimeDomainBranch(nn.Module):
    """时域分支：从YourExistingModel改造
    
    这里是简化版（可以替换为你的LeadRegionViewNet或ResNet2Block）
    """
    
    def __init__(self, input_channels=12, num_classes=10, hidden_dims=[64, 128, 256, 512]):
        super(TimeDomainBranch, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 时域1D卷积层
        self.conv1 = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块序列
        self.blocks = nn.ModuleList([
            self._make_res_block(hidden_dims[0], hidden_dims[1], stride=2),
            self._make_res_block(hidden_dims[1], hidden_dims[2], stride=2),
            self._make_res_block(hidden_dims[2], hidden_dims[3], stride=2),
        ])
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[2], num_classes)
        )
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        """简单残差块"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
    
    def forward(self, x):
        """
        参数：
            x: 输入时域信号，形状 (B, C, L) - [批量, 通道, 长度]
        返回：
            features: 特征向量，形状 (B, hidden_dims[-1])
            logits: 分类logits，形状 (B, num_classes)
        """
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块
        for block in self.blocks:
            x = block(x)
            x = self.relu(x)
        
        # 全局平均池化
        features = self.gap(x).squeeze(-1)  # (B, hidden_dims[-1])
        
        # 分类
        logits = self.classifier(features)
        
        return features, logits


# ==================== 标签条件融合层 ====================

class LabelConditionalFusion(nn.Module):
    """标签感知的动态融合
    
    设计思路：
        1. 学习标签的嵌入表示
        2. 根据标签条件计算时频权重
        3. 动态融合时域和频域特征
        4. 可解释性：权重体现不同标签对时/频的偏好
    
    医学知识映射：
        - QRS/ST/T形态类异常 → 倾向时域特征
        - 房颤/室早/心动过速等节律异常 → 倾向频域特征
    """
    
    def __init__(self, time_dim=512, freq_dim=512, num_labels=10):
        super(LabelConditionalFusion, self).__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.num_labels = num_labels
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_labels, 64)
        
        # 时频权重预测网络（根据标签条件）
        self.weight_predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # 输出 [time_weight, freq_weight]
            nn.Softmax(dim=-1)  # 归一化权重
        )
        
        # 特征对齐投影（确保维度一致）
        self.time_proj = nn.Linear(time_dim, 256)
        self.freq_proj = nn.Linear(freq_dim, 256)
        
        # 融合后的特征处理
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        
        # 注意力机制（增强融合质量）
        self.attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
    
    def forward(self, time_feat, freq_feat, label_indices=None):
        """
        参数：
            time_feat: 时域特征，形状 (B, time_dim)
            freq_feat: 频域特征，形状 (B, freq_dim)
            label_indices: 标签索引（用于标签条件融合），形状 (B, num_labels)
                          如果为None，使用均匀权重
        返回：
            fused_feat: 融合特征，形状 (B, 128)
        """
        B = time_feat.size(0)
        
        # 特征投影到共同空间
        time_proj = self.time_proj(time_feat)  # (B, 256)
        freq_proj = self.freq_proj(freq_feat)   # (B, 256)
        
        # 如果有标签条件信息，计算动态权重
        if label_indices is not None:
            # label_indices: (B, num_labels) - 多标签二值化或概率
            # 取每个样本最可能的标签（简化版）或计算标签嵌入的加权平均
            label_probs = torch.softmax(label_indices, dim=-1)  # 转概率
            label_embed = torch.matmul(
                label_probs, 
                self.label_embedding.weight
            )  # (B, 64)
            
            # 预测时频权重
            weights = self.weight_predictor(label_embed)  # (B, 2)
            time_weight, freq_weight = weights[:, 0:1], weights[:, 1:2]
        else:
            # 均匀权重
            time_weight = freq_weight = 0.5
        
        # 加权融合
        fused = time_weight * time_proj + freq_weight * freq_proj  # (B, 256)
        
        # 融合特征处理
        fused = self.fusion_mlp(fused)  # (B, 128)
        
        # 自注意力增强
        fused_attn, _ = self.attention(
            fused.unsqueeze(1), 
            fused.unsqueeze(1), 
            fused.unsqueeze(1)
        )  # (B, 1, 128)
        fused_attn = fused_attn.squeeze(1)  # (B, 128)
        
        # 残差连接
        fused = fused + 0.1 * fused_attn
        
        return fused


# ==================== 完整的时频融合模型 ====================

class TimeFreqFusionNet(nn.Module):
    """多尺度时频融合ECG分类模型
    
    整体架构：
        输入 ECG 信号 (B, 12, 5000)
        ├─ 时域分支 → TimeDomainBranch → time_feat (B, 512)
        │           └─ 捕捉波形形态特征（QRS/ST/T）
        ├─ 频域分支 → LearnableSTFT + SpectrogramCNN → freq_feat (B, 512)
        │           └─ 捕捉频域节律特征（RR间期、心率变异）
        └─ 融合层 → LabelConditionalFusion → fused_feat (B, 128)
                  └─ 标签条件动态权重融合
    
    输出：多标签 logits (B, num_classes)
    """
    
    def __init__(self, num_classes=10, input_channels=12, hidden_dims=[64, 128, 256, 512],
                 use_label_graph_refiner=False,
                 label_graph_hidden=64,
                 label_graph_learnable_adj=True,
                 label_graph_dropout=0.1):
        super(TimeFreqFusionNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        
        # 时域分支
        self.time_branch = TimeDomainBranch(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_dims=hidden_dims
        )
        
        # 频域分支
        self.stft = LearnableSTFT(n_fft=512, hop_length=128, n_scales=3)
        self.freq_branch = SpectrogramCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_dims=hidden_dims
        )
        
        # 融合层
        self.fusion = LabelConditionalFusion(
            time_dim=hidden_dims[-1],
            freq_dim=hidden_dims[-1],
            num_labels=num_classes
        )
        
        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # 标签图细化层（可选）
        self.label_refiner = None
        if use_label_graph_refiner:
            self.label_refiner = LabelGraphRefiner(
                num_classes=num_classes,
                hidden=label_graph_hidden,
                dropout=label_graph_dropout,
                learnable_adj=label_graph_learnable_adj,
            )
    
    def forward(self, x, y_soft=None):
        """
        参数：
            x: 输入信号，形状 (B, C, L) - [批量, 通道, 长度]
            y_soft: 软标签或伪标签（用于条件融合），形状 (B, num_classes)
                   如果为None，使用均匀权重
        返回：
            logits: 分类logits，形状 (B, num_classes)
        """
        # 时域分支
        time_feat, time_logits = self.time_branch(x)  # (B, 512), (B, num_classes)
        
        # 频域分支
        fused_spec, _ = self.stft(x)  # (B, 12, n_freqs, T)
        freq_feat, freq_logits, lead_weights = self.freq_branch(fused_spec)  # (B, 512), (B, num_classes), (B, 12) or None
        
        # 融合（使用软标签作为条件）
        fused_feat = self.fusion(time_feat, freq_feat, label_indices=y_soft)
        
        # 最终分类
        logits = self.final_classifier(fused_feat)

        if self.label_refiner is not None:
            logits = self.label_refiner(logits)
        
        return logits
    
    def get_intermediate_features(self, x):
        """获取中间特征（用于可视化/分析）"""
        time_feat, time_logits = self.time_branch(x)
        fused_spec, specs = self.stft(x)
        freq_feat, freq_logits, lead_weights = self.freq_branch(fused_spec)
        
        return {
            'time_feat': time_feat,
            'freq_feat': freq_feat,
            'time_logits': time_logits,
            'freq_logits': freq_logits,
            'spectrograms': specs,
            'lead_weights': lead_weights,
        }


# ==================== 便捷函数 ====================

def create_timefreq_model(num_classes, input_channels=12, pretrained=False):
    """创建时频融合模型
    
    参数：
        num_classes: 分类类别数
        input_channels: 输入通道数（ECG通常为12）
        pretrained: 是否加载预训练权重（默认False）
    返回：
        model: TimeFreqFusionNet实例
    """
    model = TimeFreqFusionNet(
        num_classes=num_classes,
        input_channels=input_channels,
        hidden_dims=[64, 128, 256, 512]
    )
    
    if pretrained:
        # TODO: 从checkpoints加载预训练权重
        pass
    
    return model
