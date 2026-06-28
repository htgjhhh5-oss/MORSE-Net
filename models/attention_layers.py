"""
注意力机制模块集合
包含：SENet（通道注意力）、CoordAtt（坐标注意力）、CBAM（通道-空间注意力）
      LeadAttention（导联注意力，用于2D频谱图的通道注意力）
用于增强 CNN 的特征表示能力，通过动态调整不同通道与空间位置的权重
"""

import torch
import torch.nn as nn


# ============================================================================
# SENet (Squeeze-and-Excitation Networks) - 通道注意力机制
# 论文：Hu et al., 2018 "Squeeze-and-Excitation Networks"
# 原理：通过全局平均池化压缩空间信息，用 FC 层学习通道间的相关性，
#       再用 sigmoid 归一化后作为通道权重，对原特征逐通道加权
# ============================================================================
class SELayer(nn.Module):
    """
    SENet 注意力层
    
    参数：
        planes (int): 输入通道数，也是输出通道数
    
    输入张量形状：[B, C, T]  (B=batch_size, C=通道数, T=时间/序列长度)
    输出张量形状：[B, C, T]  (与输入相同，仅改变通道权重)
    
    工作流程：
        1. 全局平均池化：[B, C, T] → [B, C, 1]  (对每个通道的时间维度取平均)
        2. 降维 FC：[B, C, 1] → [B, C/16, 1]  (通道数降低16倍以降低计算量)
        3. ReLU 激活：引入非线性
        4. 升维 FC：[B, C/16, 1] → [B, C, 1]  (恢复原通道数)
        5. Sigmoid：将权重压缩到 [0, 1] 范围
        6. 逐通道相乘：原特征 × 注意力权重
    """
    def __init__(self, planes):
        super(SELayer, self).__init__()
        # ReLU 激活函数，inplace=True 表示直接修改输入张量以节省内存
        self.relu = nn.ReLU(inplace=True)
        # 全局平均池化：将 [B, C, T] 压缩为 [B, C, 1]，即每个通道取平均值
        self.GAP = nn.AdaptiveAvgPool1d(1)
        # 第一个全连接层：降维，通道数从 planes → planes/16
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        # 第二个全连接层：升维，通道数从 planes/16 → planes
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        # Sigmoid 函数：将权重映射到 [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        参数：
            x [torch.Tensor]: 输入特征 [B, C, T]
        
        返回：
            out [torch.Tensor]: 加权后的特征 [B, C, T]
        """
        # 保存原始输入，用于最后的残差相乘
        original_out = x
        
        # 步骤 1: 全局平均池化，压缩时间维度
        # [B, C, T] → [B, C, 1]
        out = self.GAP(x)
        
        # 步骤 2: 展平为 2D 张量以适配全连接层
        # [B, C, 1] → [B, C]
        out = out.view(out.size(0), -1)
        
        # 步骤 3: 通过降维 FC 层 + ReLU 引入非线性
        # [B, C] → [B, C/16]
        out = self.fc1(out)
        out = self.relu(out)
        
        # 步骤 4: 通过升维 FC 层恢复通道数
        # [B, C/16] → [B, C]
        out = self.fc2(out)
        
        # 步骤 5: Sigmoid 激活得到 [0,1] 范围的权重
        out = self.sigmoid(out)
        
        # 步骤 6: reshape 回 3D 张量便于与输入做逐元素乘法
        # [B, C] → [B, C, 1]
        out = out.view(out.size(0), out.size(1), 1)
        
        # 步骤 7: 原特征与注意力权重逐通道相乘（广播）
        # [B, C, T] × [B, C, 1] → [B, C, T]
        out = out * original_out
        
        return out


# ============================================================================
# CoordAtt (Coordinate Attention) - 坐标注意力机制
# 论文：Hou et al., 2021 "Coordinate Attention for Efficient Mobile Networks"
# 原理：分别在高度和宽度两个方向上进行注意力计算，融合了通道信息和空间位置信息
#       适配 1D 序列时，将其扩展为 2D [B, C, 1, T]，在通道和时间两个维度学习依赖
# ============================================================================

# 硬 Sigmoid 激活函数：ReLU6(x+3)/6，值域为 [0, 1]
# 相比标准 Sigmoid 计算量更小，适合移动端模型
class h_sigmoid(nn.Module):
    """
    硬 Sigmoid 激活函数
    公式：h_sigmoid(x) = ReLU6(x + 3) / 6 = min(max(x+3, 0), 6) / 6
    相比标准 Sigmoid(x) = 1/(1+exp(-x))，计算更高效（无指数运算）
    
    参数：
        inplace (bool): 是否在原张量上修改（节省内存）
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        # ReLU6：max(min(x, 6), 0)，限制输出在 [0, 6]
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        # 先加 3 使得中心点从 0 移到 3，再除以 6 归一化到 [0, 1]
        return self.relu(x + 3) / 6


# 硬 Swish 激活函数：x × h_sigmoid(x)
# Swish 比 ReLU 更平滑，硬版本计算效率更高
class h_swish(nn.Module):
    """
    硬 Swish 激活函数
    公式：h_swish(x) = x × h_sigmoid(x)
    优势：比 ReLU 更光滑的激活曲线，但计算量小于标准 Swish(x × sigmoid(x))
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    坐标注意力机制（Coordinate Attention）
    
    针对 1D ECG 信号的适配：
    - 输入 [B, C, T] 序列，先 unsqueeze 为 [B, C, 1, T]（伪 2D）
    - 在"高度"方向（通道维）和"宽度"方向（时间维）分别学习注意力
    - 最后输出回 [B, C, T]
    
    原理：
    1. 沿高度方向全局平均池化：获取时间维度上的统计信息（每个位置的平均特征）
    2. 沿宽度方向保留原张量（或池化）：获取通道维度的特征
    3. 融合两个方向的信息 → 经 Conv1d + BN + h_swish 处理
    4. 分离回两个方向，分别生成高度注意力和宽度注意力
    5. 与原特征做逐元素乘法融合
    
    参数：
        inp (int): 输入通道数
        oup (int): 输出通道数（通常等于 inp）
        reduction (int): 降维比例，默认 16 倍
    
    输入张量形状：[B, C, T]
    输出张量形状：[B, C, T]
    """
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        
        # 沿高度方向（假设为通道维度的第二维）全局平均池化
        # 保留高度，压缩宽度 → 获得每个高度位置的全局特征
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 本代码中注释掉了沿宽度方向的池化，改为使用原张量的转置
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 中间通道数，至少为 8，通常为 inp/reduction
        mip = max(8, inp // reduction)

        # 第一个卷积层：压缩通道信息
        # [B, inp, h, w] → [B, mip, h, w]
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # 批归一化
        self.bn1 = nn.BatchNorm2d(mip)
        # 硬 Swish 激活：更高效的非线性变换
        self.act = h_swish()

        # 高度方向注意力卷积：[B, mip, h, w] → [B, oup, h, w]
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # 宽度方向注意力卷积：[B, mip, h, w] → [B, oup, h, w]
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播
        
        参数：
            x [torch.Tensor]: 输入特征 [B, C, T]
        
        返回：
            out [torch.Tensor]: 注意力加权后的特征 [B, C, T]
        """
        # 将 1D 序列扩展为伪 2D 格式
        # [B, C, T] → [B, C, 1, T]  (增加第 3 维，使其变为 2D 张量)
        x = x.unsqueeze(2)
        
        # 保存原始输入用于残差相乘
        identity = x
        
        # 获取张量维度
        n, c, h, w = x.size()  # n=batch, c=channel, h=1, w=time_steps
        
        # ========== 高度方向（h 方向）处理 ==========
        # 沿宽度方向全局平均池化：保留高度信息，压缩宽度
        # [B, C, 1, T] → [B, C, 1, 1]
        x_h = self.pool_h(x)
        
        # ========== 宽度方向（w 方向）处理 ==========
        # 通过转置把宽度变为高度，保留原宽度特征分布
        # [B, C, 1, T] → [B, C, T, 1]  (维度 2 和 3 交换)
        x_w = x.permute(0, 1, 3, 2)
        
        # ========== 融合高度和宽度信息 ==========
        # 沿高度维度拼接两个方向的信息
        # [B, C, 1, 1] + [B, C, T, 1] → [B, C, 1+T, 1]
        y = torch.cat([x_h, x_w], dim=2)
        
        # 通过卷积 + BN + h_swish 处理融合的信息
        # [B, C, 1+T, 1] → [B, mip, 1+T, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # ========== 分离高度和宽度的注意力 ==========
        # 从融合特征中分离出高度和宽度部分
        # [B, mip, 1+T, 1] → [B, mip, 1, 1] + [B, mip, T, 1]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        
        # 将宽度方向转置回原顺序
        # [B, mip, T, 1] → [B, mip, 1, T]
        x_w = x_w.permute(0, 1, 3, 2)
        
        # ========== 生成注意力权重 ==========
        # 高度方向注意力：[B, mip, 1, 1] → [B, oup, 1, 1] → sigmoid → [0,1]
        a_h = self.conv_h(x_h).sigmoid()
        
        # 宽度方向注意力：[B, mip, 1, T] → [B, oup, 1, T] → sigmoid → [0,1]
        a_w = self.conv_w(x_w).sigmoid()
        
        # ========== 应用注意力权重 ==========
        # 原特征与两个方向的注意力权重逐元素相乘
        # [B, C, 1, T] × [B, C, 1, 1] × [B, C, 1, T] → [B, C, 1, T]
        out = identity * a_w * a_h
        
        # 压缩回 1D 序列格式
        # [B, C, 1, T] → [B, C, T]
        out = out.squeeze(2)
        
        return out



# ============================================================================
# CBAM (Convolutional Block Attention Module) - 通道-空间注意力模块
# 论文：Woo et al., 2018 "CBAM: Convolutional Block Attention Module"
# 原理：串联使用通道注意力和空间注意力，两个机制互相补充
#       通道注意力：学习哪些通道重要
#       空间注意力：学习特征图中哪些空间位置重要
# ============================================================================

# 通道注意力子模块
class ChannelAttention(nn.Module):
    """
    通道注意力机制（Channel Attention Module）
    
    在 CBAM 中的角色：
    - 输入：[B, C, T]
    - 处理：分别用最大池化和平均池化获取全局特征，经 FC 层处理后相加
    - 输出：通道权重 [B, C, 1]
    
    工作流程：
    1. 最大池化 + 平均池化：分别提取特征图中的最大值和平均值
    2. 通过 FC 层（用 Conv1d 实现）学习通道间的非线性关系
    3. 两个路径结果相加后用 Sigmoid 归一化到 [0, 1]
    
    参数：
        in_planes (int): 输入通道数
        ratio (int): 降维比例，默认 16 倍（与 SENet 相同）
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        # 全局平均池化：[B, C, T] → [B, C, 1]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全局最大池化：[B, C, T] → [B, C, 1]
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 降维卷积：[B, C, 1] → [B, C/ratio, 1]
        # 用 Conv1d 而非 Linear 可以保持 3D 张量形状，避免 reshape
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        # ReLU 激活函数
        self.relu1 = nn.ReLU()
        # 升维卷积：[B, C/ratio, 1] → [B, C, 1]
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        # Sigmoid 激活：将权重映射到 [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        参数：
            x [torch.Tensor]: 输入特征 [B, C, T]
        
        返回：
            [torch.Tensor]: 通道注意力权重 [B, C, 1]
        """
        # 路径 1: 平均池化 → 降维 → ReLU → 升维
        # [B, C, T] → [B, C, 1] → [B, C/ratio, 1] → [B, C, 1]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        
        # 路径 2: 最大池化 → 降维 → ReLU → 升维
        # [B, C, T] → [B, C, 1] → [B, C/ratio, 1] → [B, C, 1]
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        
        # 融合两个路径的信息：加法融合
        # [B, C, 1] + [B, C, 1] → [B, C, 1]
        out = avg_out + max_out
        
        # 用 Sigmoid 将权重映射到 [0, 1]
        return self.sigmoid(out)


# 空间注意力子模块
class SpatialAttention(nn.Module):
    """
    空间注意力机制（Spatial Attention Module）
    
    在 CBAM 中的角色：
    - 输入：[B, C, T]
    - 处理：沿通道维度计算最大值和平均值，拼接后用卷积学习空间权重
    - 输出：空间权重 [B, 1, T]
    
    工作流程：
    1. 沿通道维度分别计算最大值和平均值：每个时间步的统计特征
    2. 拼接两个统计特征：通道维度变为 2
    3. 用 Conv1d(2, 1, ...) 压缩通道，生成每个时间位置的权重
    4. Sigmoid 归一化到 [0, 1]
    
    参数：
        kernel_size (int): 卷积核大小，默认 1（保持位置信息）
        padding (int): 填充大小，默认 0
    """
    def __init__(self, kernel_size=1, padding=0):
        super(SpatialAttention, self).__init__()
        
        # 空间卷积层：融合最大值和平均值信息
        # 输入通道数为 2（因为后续会 concat 最大值和平均值）
        # 输出通道数为 1（生成单一的空间权重图）
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        # Sigmoid 激活：将权重映射到 [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        
        参数：
            x [torch.Tensor]: 输入特征 [B, C, T]
        
        返回：
            [torch.Tensor]: 空间注意力权重 [B, 1, T]
        """
        # 路径 1: 沿通道维度计算平均值
        # [B, C, T] → [B, 1, T]  (keepdim=True 保持维度数，避免自动压缩)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 路径 2: 沿通道维度计算最大值
        # [B, C, T] → [B, 1, T]  (torch.max 返回 (values, indices) 元组)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 融合两个路径：沿通道维度拼接
        # [B, 1, T] 和 [B, 1, T] → [B, 2, T]
        x = torch.cat([avg_out, max_out], dim=1)
        
        # 用卷积层学习空间权重
        # [B, 2, T] → [B, 1, T]
        x = self.conv1(x)
        
        # Sigmoid 激活得到 [0, 1] 范围的权重
        return self.sigmoid(x)


# CBAM 完整模块
class CBAM(nn.Module):
    """
    卷积块注意力模块（Convolutional Block Attention Module）
    
    完整的 CBAM 流程：
    1. 通道注意力：识别哪些通道重要，生成通道权重 [B, C, 1]
    2. 通道加权：特征 × 通道权重
    3. 空间注意力：识别哪些空间位置重要，生成空间权重 [B, 1, T]
    4. 空间加权：通道加权后的特征 × 空间权重
    5. 最终输出：经过双重注意力加权的特征
    
    这样的串联设计使得模型既能学习通道间的重要性，又能学习空间位置的重要性。
    
    参数：
        channel (int): 输入通道数
    
    输入张量形状：[B, C, T]
    输出张量形状：[B, C, T]  (形状不变，仅改变权重)
    """
    def __init__(self, channel):
        super(CBAM, self).__init__()
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channel)
        # 空间注意力模块
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        """
        前向传播
        
        参数：
            x [torch.Tensor]: 输入特征 [B, C, T]
        
        返回：
            out [torch.Tensor]: 双重注意力加权后的特征 [B, C, T]
        """
        # 步骤 1: 通道注意力加权
        # [B, C, T] × [B, C, 1] → [B, C, T]
        out = self.channel_attention(x) * x
        
        # 步骤 2: 空间注意力加权
        # [B, C, T] × [B, 1, T] → [B, C, T]
        out = self.spatial_attention(out) * out
        
        return out


# ============================================================================
# CrossModalAttention - 跨模态双向交叉注意力 (Mid-Fusion)
# 用于时域-频域中层融合：在特征图级别交换信息，而非仅在最终特征上做后融合
# ============================================================================

class CrossModalAttention(nn.Module):
    """跨模态双向交叉注意力模块 (Bidirectional Cross-Modal Attention)

    核心思想：让时域和频域特征在中间层（池化前）就进行信息交换
      - 频域 Query → 时域 Key/Value：频域发现的"节律异常"去时域中寻找对应的"异常波形"
      - 时域 Query → 频域 Key/Value：时域的波形形态特征去频域中寻找对应的频率成分

    每个方向采用 Pre-Norm Transformer 块：CrossAttn → LN → FFN → LN

    输入：
        time_seq: (B, L_t, D) - 时域序列特征（多视图Conv特征图降采样后拼接）
        freq_seq: (B, L_f, D) - 频域序列特征（STFT谱图经2D-CNN编码）
    输出：
        time_enhanced: (B, L_t, D) - 经频域信息增强的时域特征
        freq_enhanced: (B, L_f, D) - 经时域信息增强的频域特征

    参数：
        d_model (int): 特征维度（默认128，与视图特征维度一致）
        nhead (int): 注意力头数（默认4）
        dropout (float): Dropout 比例（默认0.1）
        residual_scale (float): 残差连接缩放因子（默认0.5，控制交叉信息注入强度）
    """

    def __init__(self, d_model=128, nhead=4, dropout=0.1, residual_scale=0.5):
        super(CrossModalAttention, self).__init__()
        self.residual_scale = residual_scale

        # ---- 路径1: 频域 Query → 时域 Key/Value ----
        # 频域特征作为 Query，去时域中寻找对应的波形位置
        self.freq2time_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_f2t = nn.LayerNorm(d_model)
        self.ffn_freq = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn_freq = nn.LayerNorm(d_model)

        # ---- 路径2: 时域 Query → 频域 Key/Value ----
        # 时域特征作为 Query，去频域中寻找对应的频率成分
        self.time2freq_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_t2f = nn.LayerNorm(d_model)
        self.ffn_time = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn_time = nn.LayerNorm(d_model)

    def forward(self, time_seq, freq_seq):
        """双向交叉注意力前向传播

        参数：
            time_seq: (B, L_t, D) - 时域序列 token
            freq_seq: (B, L_f, D) - 频域序列 token
        返回：
            time_enhanced: (B, L_t, D) - 增强后的时域序列
            freq_enhanced: (B, L_f, D) - 增强后的频域序列
        """
        # ---- 频域 Query → 时域 Key/Value ----
        # freq 向 time 提问："哪些时域位置与我检测到的频率模式相关？"
        freq_attn_out, _ = self.freq2time_attn(
            query=freq_seq, key=time_seq, value=time_seq
        )
        freq_enhanced = self.norm_f2t(freq_seq + self.residual_scale * freq_attn_out)
        freq_enhanced = self.norm_ffn_freq(freq_enhanced + self.ffn_freq(freq_enhanced))

        # ---- 时域 Query → 频域 Key/Value ----
        # time 向 freq 提问："我检测到的波形异常对应哪些频率成分？"
        time_attn_out, _ = self.time2freq_attn(
            query=time_seq, key=freq_seq, value=freq_seq
        )
        time_enhanced = self.norm_t2f(time_seq + self.residual_scale * time_attn_out)
        time_enhanced = self.norm_ffn_time(time_enhanced + self.ffn_time(time_enhanced))

        return time_enhanced, freq_enhanced


# ============================================================================
# LeadAttention - 导联注意力机制 (Lead Attention for 2D Spectrogram)
# 针对ECG多导联频谱图设计的通道注意力模块
# 核心动机：不同心脏疾病在不同导联上的频域表现差异显著
#   - 下壁心梗：II, III, aVF 导联的 ST 段抬高 → 低频能量异常
#   - 前壁心梗：V1-V4 导联的 Q 波加深 → 特征频段变化
#   - 房颤：全导联 P 波消失 → 高频不规则活动
# 该模块让网络自适应地放大含关键疾病频谱的导联通道，抑制无关导联的噪声
# ============================================================================

class LeadAttention(nn.Module):
    """导联注意力机制 (Lead Attention) - 2D频谱图的通道注意力

    在 SpectrogramCNN 处理之前，对 12 导联的 STFT 频谱图施加通道注意力，
    让网络学习哪些导联对当前样本的分类最重要。

    设计特点：
        1. 双路径特征聚合：全局平均池化 + 全局最大池化（CBAM 风格），
           比单一池化更好地捕捉导联特征的统计量和显著性
        2. 共享 FC 瓶颈层：两路径共享参数，减少过拟合风险
        3. 频率-时间感知增强（可选）：除通道注意力外，额外学习频率-时间
           维度的空间权重，识别关键的频段和时间段

    参数：
        num_leads (int): 导联数量，默认 12（标准12导联ECG）
        reduction (int): FC 瓶颈降维比例，默认 4
            - 12 导联 / 4 = 3 维中间层，足够建模导联间关系
        use_spectral_spatial (bool): 是否启用频率-时间空间注意力，默认 True

    输入张量形状：(B, C, F, T) - [批量, 导联数, 频率bins, 时间帧]
    输出张量形状：(B, C, F, T) - 导联注意力加权后的频谱图
    """

    def __init__(self, num_leads=12, reduction=4, use_spectral_spatial=True, spatial_kernel_size=7):
        super(LeadAttention, self).__init__()
        self.num_leads = num_leads
        self.use_spectral_spatial = use_spectral_spatial
        self.spatial_kernel_size = spatial_kernel_size

        # --- 导联通道注意力 (Channel Attention on Leads) ---
        mid_channels = max(2, num_leads // reduction)

        # 全局池化：(B, C, F, T) → (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 FC 瓶颈（avg 和 max 路径共享权重，减少参数量）
        # 使用 Conv2d(1×1) 等效于 Linear，但保持 4D 张量形状，避免 reshape
        self.shared_fc = nn.Sequential(
            nn.Conv2d(num_leads, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_leads, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

        # --- 频率-时间空间注意力 (Spectral-Spatial Attention，可选) ---
        # 在导联注意力之后，进一步识别重要的频率-时间区域
        if self.use_spectral_spatial:
            _pad = spatial_kernel_size // 2  # 保证 same padding
            self.spatial_conv = nn.Sequential(
                # 输入通道 = 2（avg + max 沿通道维度），输出通道 = 1
                nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=_pad, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        """
        前向传播

        参数：
            x: (B, C, F, T) - 多导联频谱图
                B: batch size
                C: 导联数（12）
                F: 频率 bins
                T: 时间帧数
        返回：
            out: (B, C, F, T) - 导联注意力加权后的频谱图
            lead_weights: (B, C) - 每个导联的注意力权重（可用于可视化分析）
        """
        # ========== Step 1: 导联通道注意力 ==========
        # 全局平均池化路径：对每个导联的 (F, T) 频谱取全局平均
        # (B, C, F, T) → (B, C, 1, 1)
        avg_out = self.shared_fc(self.avg_pool(x))

        # 全局最大池化路径：对每个导联的 (F, T) 频谱取全局最大值
        # 最大池化捕捉导联中最显著的频谱峰值
        # (B, C, F, T) → (B, C, 1, 1)
        max_out = self.shared_fc(self.max_pool(x))

        # 两路融合 → sigmoid → 导联权重
        # (B, C, 1, 1)
        lead_attn = self.sigmoid(avg_out + max_out)

        # 导联注意力权重（用于可视化）
        # (B, C, 1, 1) → (B, C)
        lead_weights = lead_attn.squeeze(-1).squeeze(-1)

        # 导联重标定：放大重要导联，抑制无关导联
        # (B, C, F, T) × (B, C, 1, 1) → (B, C, F, T)
        out = x * lead_attn

        # ========== Step 2: 频率-时间空间注意力（可选） ==========
        if self.use_spectral_spatial:
            # 沿导联（通道）维度计算统计量
            # 平均值：每个 (f, t) 位置上 12 个导联的平均响应
            avg_spatial = torch.mean(out, dim=1, keepdim=True)  # (B, 1, F, T)
            # 最大值：每个 (f, t) 位置上最显著的导联响应
            max_spatial, _ = torch.max(out, dim=1, keepdim=True)  # (B, 1, F, T)

            # 拼接后通过卷积生成空间注意力图
            spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # (B, 2, F, T)
            spatial_attn = self.spatial_conv(spatial_input)  # (B, 1, F, T)

            # 空间注意力重标定：放大关键频率-时间区域
            out = out * spatial_attn

        return out, lead_weights

