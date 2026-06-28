# -*- coding: utf-8 -*-
'''
@time: 2021/4/17 20:14
@ author: ysx
@ description: 多视图ECG分类模型（基于Res2Block+CoordAtt注意力+自适应权重融合）
               核心设计：12导联ECG信号划分为6个视图，通过独立分支提取特征后加权融合
               适配场景：单标签ECG分类（可修改为多标签，见注释说明）
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入坐标注意力模块（CoordAtt）：增强空间-通道关联特征捕捉
from models.attention_layers import CoordAtt




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

# ---------------------- 激活函数模块 ----------------------
class Mish(nn.Module):
    """Mish激活函数：x * tanh(softplus(x))
    优势：相比ReLU缓解梯度消失，在时序信号中保留更多细节特征
    适配场景：ECG信号的非线性特征提取
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 计算逻辑：softplus(x)将输入映射到正区间，tanh增强非线性，最后与原输入相乘
        return x * (torch.tanh(F.softplus(x)))


# ---------------------- 工具卷积层 ----------------------
def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积层（1D）：用于通道数调整和维度压缩/扩张
    参数说明：
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 步幅（默认1，不改变长度）
    作用：在不改变特征图长度的前提下，调整通道数，降低计算量
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


# ---------------------- 改进型残差块（Res2Block） ----------------------
class Res2Block(nn.Module):
    """Res2Block残差块（1D适配）：多分支特征融合+注意力增强
    核心改进：将输入特征拆分为多个分支，逐步融合特征，提升感受野和特征表达能力
    适配场景：ECG时序信号的深层特征提取，保留局部细节和全局依赖
    """
    expansion = 1  # 残差块输出通道数扩张系数（1表示不扩张）

    def __init__(self, inplanes, planes, kernel_size=5, stride=1, downsample=None, groups=1, base_width=26,
                 dilation=1, scale=4, first_block=True, norm_layer=nn.BatchNorm1d,
                 atten=True):
        """
        参数说明：
            inplanes: 输入通道数
            planes: 中间特征通道数
            kernel_size: 卷积核大小（默认5，适配ECG信号的局部特征）
            stride: 步幅（默认1，stride=2时实现下采样）
            downsample:  shortcut路径的下采样模块（当输入输出维度不匹配时使用）
            groups: 分组卷积参数（默认1，即普通卷积）
            base_width: 基础宽度（用于计算分支通道数）
            dilation: 空洞卷积系数（默认1，不使用空洞卷积）
            scale: 分支数量（默认4，拆分为4个分支融合特征）
            first_block: 是否为该层第一个残差块（默认True，第一个块使用平均池化）
            norm_layer: 归一化层（默认BatchNorm1d，时序数据适配）
            atten: 是否添加注意力机制（默认True，启用CoordAtt）
        """
        super(Res2Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d  # 默认为1D批量归一化

        # 计算每个分支的通道数：基于base_width和groups调整
        width = int(planes * (base_width / 64.)) * groups

        self.atten = atten  # 是否启用注意力标记
        self.scale = scale  # 分支数量
        self.first_block = first_block  # 是否为该层第一个块

        # 1x1卷积：将输入通道数调整为 分支数×单分支通道数（width×scale）
        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)  # 批量归一化：加速训练，缓解梯度消失

        # 构建多分支卷积：分支数=scale-1（最后一个分支为原特征或池化后特征）
        nb_branches = max(scale, 2) - 1
        if first_block:
            # 第一个块的最后一个分支使用平均池化（下采样+特征平滑）
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)

        # 初始化分支卷积和归一化层
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=1, bias=False, dilation=1)
            for _ in range(nb_branches)
        ])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])

        # 1x1卷积：将多分支融合后的特征通道数还原为 planes×expansion
        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)  # 输出归一化
        self.relu = Mish()  # 激活函数

        # 注意力模块：CoordAtt（坐标注意力，捕捉空间-通道关联）
        if self.atten is True:
            self.attention = CoordAtt(planes * self.expansion, planes * self.expansion)
        else:
            self.attention = None

        # shortcut路径：当输入输出维度不匹配时（stride≠1或通道数不同），用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        """前向传播流程：多分支特征提取→融合→注意力增强→残差连接
        输入x: 形状为 (batch_size, inplanes, seq_len) （批量大小，输入通道数，信号长度）
        输出out: 形状为 (batch_size, planes×expansion, seq_len//stride)
        """
        # 第一步：1x1卷积调整通道数 + 激活 + 归一化
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)  # 注意：这里先激活后归一化（bn reverse设计）

        # 第二步：多分支特征提取与融合
        xs = torch.chunk(out, self.scale, dim=1)  # 按通道维度拆分特征（拆分为scale个分支）
        y = 0  # 初始化分支输出
        for idx, conv in enumerate(self.convs):
            # 分支融合逻辑：前一个分支的输出与当前分支输入相加
            if self.first_block:
                y = xs[idx]  # 第一个块：直接使用当前分支特征
            else:
                y += xs[idx]  # 非第一个块：累加前序分支特征（残差融合）

            # 分支卷积 + 归一化 + 激活
            y = conv(y)
            y = self.relu(self.bns[idx](y))

            # 拼接分支输出（第一个分支直接作为out，后续分支拼接）
            out = torch.cat((out, y), 1) if idx > 0 else y

        # 第三步：添加最后一个分支（原特征或池化后特征）
        if self.scale > 1:
            if self.first_block:
                # 第一个块：最后一个分支用平均池化
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                # 非第一个块：直接使用最后一个分支的原特征
                out = torch.cat((out, xs[len(self.convs)]), 1)

        # 第四步：1x1卷积还原通道数 + 归一化
        out = self.conv3(out)
        out = self.bn3(out)

        # 第五步：注意力增强（可选）
        if self.atten:
            out = self.attention(out)

        # 第六步：残差连接（shortcut路径 + 主路径特征）
        out += self.shortcut(x)
        out = self.relu(out)  # 最终激活

        return out


# ---------------------- 基础单视图模型 ----------------------
class TemporalRes2NetBackbone(nn.Module):
    """基础ECG特征提取网络（单视图）：Conv1d+Res2Block+全局平均池化
    作用：作为多视图模型的基础分支，处理单个视图的ECG信号
    """

    def __init__(self, num_classes=5, input_channels=12, single_view=False):
        """
        参数说明：
            num_classes: 分类类别数（默认5，需根据实际数据集调整）
            input_channels: 输入通道数（默认12，对应12导联；单视图时为单个/多个导联组合）
            single_view: 是否为单视图分支（默认False：输出分类结果；True：输出特征向量，用于多视图融合）
        """
        super().__init__()
        self.single_view = single_view  # 标记是否为多视图中的分支

        # 第一层卷积：将输入信号映射到64维特征， kernel_size=25（捕捉ECG信号的基础形态特征）
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=25, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)  # 批量归一化
        self.relu = Mish()  # 激活函数

        # 残差块1：输入64维→输出128维，stride=2（下采样，减半信号长度），启用注意力
        self.layer1 = Res2Block(inplanes=64, planes=128, kernel_size=15, stride=2, atten=True)
        # 残差块2：输入128维→输出128维，stride=2（再次下采样），启用注意力
        self.layer2 = Res2Block(inplanes=128, planes=128, kernel_size=15, stride=2, atten=True)

        # 全局平均池化：将时序特征压缩为1维向量（batch_size, 128, 1）→（batch_size, 128）
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 分类头：仅当single_view=False时启用（单视图直接分类）
        if not self.single_view:
            self.fc = nn.Linear(128, num_classes)  # 128维特征→num_classes类输出

        # 初始化权重：卷积层用kaiming_normal，BN层权重设为1、偏置设为0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_seq=False, return_intermediate=False):
        """前向传播流程：卷积提取基础特征→残差块深层特征→池化→分类/特征输出
        输入x: 形状为 (batch_size, input_channels, seq_len)
        输出：
            single_view=False时为 (batch_size, num_classes)；True时为 (batch_size, 128)
            若 return_seq=True，额外返回池化前的序列特征 (batch_size, 128, L'')
        """
        # 基础特征提取
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        # 深层特征提取（两次残差块+下采样）
        output = self.layer1(output)
        output = self.layer2(output)

        # 保存池化前的序列特征（用于跨模态中融合 Cross-Modal Mid-Fusion）
        seq_features = None
        if return_seq or return_intermediate:
            seq_features = output  # (batch_size, 128, L'')

        # 全局平均池化：压缩时序维度
        output = self.avgpool(output)

        # 展平特征：(batch_size, 128, 1) → (batch_size, 128)
        fused_feat = output.view(output.size(0), -1)

        # 分类输出（单视图模式）或特征输出（多视图分支模式）
        output = fused_feat
        if not self.single_view:
            output = self.fc(output)

        if return_intermediate:
            intermediate = {
                'fused_feat': fused_feat,
                'seq_features': seq_features,
                'view_features': [],
                'representation_weights': [],
                'fuse_weights': [],
                'view_names': ['single_view_12lead'],
            }
            return output, intermediate

        if return_seq:
            return output, seq_features
        return output


class TwelveLeadBaselineNet(TemporalRes2NetBackbone):
    """Explicit 12-lead single-view classifier alias for distillation experiments."""

    def __init__(self, num_classes=5):
        super().__init__(num_classes=num_classes, input_channels=12, single_view=False)


# ---------------------- 自适应权重融合模块 ----------------------
class AdaptiveViewWeight(nn.Module):
    """自适应权重生成模块：为每个视图的特征向量分配动态权重
    核心逻辑：根据特征向量的内容，通过全连接层学习权重（0~1之间），实现视图重要性自适应调整
    """

    def __init__(self, plances=32):
        """
        参数说明：
            plances: 输入特征维度（默认32，需与视图分支输出特征维度一致）
        """
        super().__init__()
        # 全连接层：将特征向量映射为1维权重
        self.fc = nn.Linear(plances, 1)
        self.sig = nn.Sigmoid()  # Sigmoid激活：将权重归一化到[0,1]

    def forward(self, x):
        """前向传播：特征→权重
        输入x: 形状为 (batch_size, plances) （单个视图的特征向量）
        输出out: 形状为 (batch_size, 1) （每个样本的自适应权重）
        """
        out = self.fc(x)  # 特征→1维分数
        out = self.sig(out)  # 归一化到[0,1]
        return out


# ---------------------- 多视图融合模型（核心） ----------------------
class LeadRegionViewNet(nn.Module):
    """6视图ECG分类模型：将12导联ECG划分为6个视图，通过独立分支提取特征后自适应加权融合
    视图划分逻辑（基于12导联的临床意义和信号相关性）：
        视图1：导联3（单导联）
        视图2：导联0 + 导联4（双导联组合）
        视图3：导联6 + 导联7（双导联组合）
        视图4：导联8 + 导联9（双导联组合）
        视图5：导联10 + 导联11（双导联组合）
        视图6：导联1 + 导联2 + 导联5（三导联组合）
    融合方式：自适应权重加权求和（每个视图的权重由AdaptiveViewWeight 学习）
    """

    def __init__(self, num_classes=5):
        """
        参数说明：
            num_classes: 分类类别数（默认5，需根据实际数据集调整，如你的CPSC2018为9类）
        """
        super().__init__()

        # 6个视图的基础分支（single_view=True：输出128维特征向量）
        self.lead_i_branch = TemporalRes2NetBackbone(input_channels=1, single_view=True)  # 视图1：1导联输入
        self.augmented_limb_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True)  # 视图2：2导联输入
        self.septal_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True)  # 视图3：2导联输入
        self.anterior_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True)  # 视图4：2导联输入
        self.lateral_precordial_branch = TemporalRes2NetBackbone(input_channels=2, single_view=True)  # 视图5：2导联输入
        self.inferior_limb_branch = TemporalRes2NetBackbone(input_channels=3, single_view=True)  # 视图6：3导联输入

        # 6个视图的自适应权重模块（输入特征维度=128，与分支输出一致）
        self.lead_i_weight = AdaptiveViewWeight(128)
        self.augmented_limb_weight = AdaptiveViewWeight(128)
        self.septal_precordial_weight = AdaptiveViewWeight(128)
        self.anterior_precordial_weight = AdaptiveViewWeight(128)
        self.lateral_precordial_weight = AdaptiveViewWeight(128)
        self.inferior_limb_weight = AdaptiveViewWeight(128)

        # 最终分类头：融合后的特征→类别概率
        self.fc = nn.Linear(128, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_intermediate=False):
        """前向传播流程：输入12导联ECG→6视图划分→分支特征提取→自适应加权融合→分类
        输入x: 形状为 (batch_size, 12, seq_len) （12导联ECG信号，seq_len为采样点数量）
        输出x_out: 形状为 (batch_size, num_classes) （每个类别的预测概率）
        """
        # 第一步：6视图信号划分 + 各视图特征提取
        outputs_view = [
            # 视图1：提取导联3（索引3），unsqueeze(1)将维度从(batch_size, seq_len)→(batch_size, 1, seq_len)
            self.lead_i_branch(x[:, 3, :].unsqueeze(1)),
            # 视图2：拼接导联0和导联4（维度：(batch_size, 2, seq_len)）
            self.augmented_limb_branch(torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)),
            # 视图3：提取导联6-7（切片操作，维度：(batch_size, 2, seq_len)）
            self.septal_precordial_branch(x[:, 6:8, :]),
            # 视图4：提取导联8-9
            self.anterior_precordial_branch(x[:, 8:10, :]),
            # 视图5：提取导联10-11
            self.lateral_precordial_branch(x[:, 10:12, :]),
            # 视图6：拼接导联1-2和导联5（维度：(batch_size, 3, seq_len)）
            self.inferior_limb_branch(torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1))
        ]  # outputs_view形状：[6个 (batch_size, 128) 的特征向量]

        # 第二步：为每个视图特征生成自适应权重
        lead_i_score = self.lead_i_weight(outputs_view[0])  # (batch_size, 1)
        augmented_limb_score = self.augmented_limb_weight(outputs_view[1])
        septal_precordial_score = self.septal_precordial_weight(outputs_view[2])
        anterior_precordial_score = self.anterior_precordial_weight(outputs_view[3])
        lateral_precordial_score = self.lateral_precordial_weight(outputs_view[4])
        inferior_limb_score = self.inferior_limb_weight(outputs_view[5])

        # 第三步：加权融合（权重×特征向量，然后求和）
        output = (lead_i_score * outputs_view[0] +
                  augmented_limb_score * outputs_view[1] +
                  septal_precordial_score * outputs_view[2] +
                  anterior_precordial_score * outputs_view[3] +
                  lateral_precordial_score * outputs_view[4] +
                  inferior_limb_score * outputs_view[5])  # 融合后形状：(batch_size, 128)

        # 第四步：最终分类（融合特征→类别概率）
        x_out = self.fc(output)

        if return_intermediate:
            intermediate = {
                'view_features': outputs_view,
                'representation_weights': [lead_i_score, augmented_limb_score, septal_precordial_score,
                                 anterior_precordial_score, lateral_precordial_score, inferior_limb_score],
                # Backward-compatible key consumed by distillation/plot scripts.
                'fuse_weights': [lead_i_score, augmented_limb_score, septal_precordial_score,
                                 anterior_precordial_score, lateral_precordial_score, inferior_limb_score],
                'fused_feat': output,
                'view_names': ['lead_I', 'augmented_limb_aVR_aVL', 'septal_V1_V2',
                              'anterior_V3_V4', 'lateral_V5_V6', 'inferior_II_III_aVF'],
            }
            return x_out, intermediate

        return x_out

    def load_state_dict(self, state_dict, *args, **kwargs):
        legacy_prefix_pairs = [
            ('MyNet1', 'lead_i_branch'),
            ('MyNet2', 'augmented_limb_branch'),
            ('MyNet3', 'septal_precordial_branch'),
            ('MyNet4', 'anterior_precordial_branch'),
            ('MyNet5', 'lateral_precordial_branch'),
            ('MyNet6', 'inferior_limb_branch'),
            ('fuse_weight_1', 'lead_i_weight'),
            ('fuse_weight_2', 'augmented_limb_weight'),
            ('fuse_weight_3', 'septal_precordial_weight'),
            ('fuse_weight_4', 'anterior_precordial_weight'),
            ('fuse_weight_5', 'lateral_precordial_weight'),
            ('fuse_weight_6', 'inferior_limb_weight'),
        ]
        state_dict = _remap_state_dict_keys(state_dict, legacy_prefix_pairs)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def get_view_weights(self, x):
        _, intermediate = self.forward(x, return_intermediate=True)
        weights = torch.cat(intermediate['representation_weights'], dim=1)
        weights = F.softmax(weights, dim=1)
        return weights.detach().cpu().numpy()

# ---------------------- 适配你的CPSC2018数据集的修改建议 ----------------------
# 1. 类别数调整：你的数据集为9类，初始化模型时设置 num_classes=9
# model = LeadRegionViewNet(num_classes=9)
#
# 2. 输入数据适配：
#    - 若使用原始12导联信号（1D）：直接输入 (batch_size, 12, seq_len)，无需修改模型
#    - 若使用你下载的图片数据集（518x518x3）：需添加图片→12导联信号的转换模块（如CNN特征提取+导联映射）
#      或修改基础分支为2D卷积（将 TemporalRes2NetBackbone 中的 Conv1d改为Conv2d，Res2Block适配2D操作）
#
# 3. 多标签支持：若需处理多标签分类（如ECG可能同时存在多种异常），修改分类头为：
# self.fc = nn.Linear(128, num_classes)
# 并在训练时使用 BCEWithLogitsLoss 损失函数


# ---------------------------------------------------------------------------
# Backward-compatible aliases.  Older experiment configs and checkpoints may
# still refer to these names; new code should prefer the explicit names above.
MyNet = TemporalRes2NetBackbone
MyNet12Lead = TwelveLeadBaselineNet
MyNet6View = LeadRegionViewNet
AdaptiveWeight = AdaptiveViewWeight
