# -*- coding: utf-8 -*-
"""
Label graph modules:
  1. LabelGraphRefiner  – logits-level GCN post-processor (original)
  2. FeatureLabelGCN    – feature-level GCN that acts on fused_feat *before*
                          the classifier head, injecting label co-occurrence
                          information directly into the high-dim feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Logits-Level (Original) ====================

class LabelGraphRefiner(nn.Module):
    """在 logits 空间做轻量 GCN 平滑（后处理）。

    修复说明（v2）：
        1. 将硬编码 0.5 替换为可学习门控（init_gate=-3.0 → sigmoid≈0.047），
           训练初期几乎不干扰 logits，随训练渐进激活。
        2. MLP 输入前加 LayerNorm，稳定不同 epoch 的 logits 尺度波动。
        3. 邻接矩阵始终做 softmax 归一化（不论可学习与否），保证行和=1。
    """

    def __init__(self, num_classes, hidden=64, dropout=0.1, learnable_adj=True,
                 init_gate=-3.0):
        super().__init__()
        init_adj = torch.eye(num_classes) + 0.1 * torch.ones(num_classes, num_classes)
        if learnable_adj:
            self.base_adj = nn.Parameter(init_adj)
        else:
            self.register_buffer("base_adj", init_adj)

        # 输入归一化：稳定 MLP 输入的尺度
        self.input_norm = nn.LayerNorm(num_classes)
        self.lin1 = nn.Linear(num_classes, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

        # 可学习门控：sigmoid(init_gate) 控制初始贡献比例
        # -3.0 → 0.047, -2.0 → 0.12, 0.0 → 0.50
        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, logits):
        # logits: (B, C)
        # 邻接矩阵始终 softmax 归一化（行和=1）
        A = F.softmax(self.base_adj, dim=-1)

        # 图传播：节点 i 聚合所有邻居 j 的信息，权重为 A[i,j]
        # 正确写法：logits @ A^T，使得 context[b,i] = sum_j A[i,j]*logits[b,j]
        # （与 FeatureLabelGCN 中 torch.matmul(A, h) 的语义一致）
        context = logits @ A.t()  # (B, C) @ (C, C) → (B, C)

        # 归一化后送入MLP
        fused = self.input_norm(context)
        fused = self.lin2(self.dropout(F.relu(self.lin1(fused))))

        # 门控残差：训练初期 gate≈0.047，几乎不干扰原始 logits
        gate = torch.sigmoid(self.gate)
        return logits + gate * fused


# ==================== Feature-Level (New) ====================

class FeatureLabelGCN(nn.Module):
    """Feature-level Label Graph Convolution Network.

    在分类头 (fc) **之前** 对融合特征施加标签共现建模：
        1. 将 fused_feat (B, D) 投影为 **每标签隐状态** (B, C, H)
        2. 两层 GCN（带可学习邻接矩阵）在标签维度传播信息
        3. 聚合回特征空间 (B, D)，以门控残差注入原始特征

    与 LabelGraphRefiner 的区别：
        - LabelGraphRefiner 在 logits(C维) 上做平滑 → "概率后修正"
        - FeatureLabelGCN 在高维特征(D维) 上做图卷积 → "语义预建模"
          每个标签节点拥有 H 维特征表示，可编码更丰富的共现语义

    参数：
        feat_dim       (int) : 输入特征维度（默认128，匹配 fused_feat）
        num_classes    (int) : 标签数（即图节点数）
        gcn_hidden     (int) : GCN 隐层维度
        num_gcn_layers (int) : GCN 层数（1 或 2）
        dropout       (float): GCN 内部 dropout
        learnable_adj (bool) : 邻接矩阵是否可学习
        init_gate     (float): 门控初始值（sigmoid 前），越小初始影响越弱
    """

    def __init__(
        self,
        feat_dim=128,
        num_classes=9,
        gcn_hidden=64,
        num_gcn_layers=2,
        dropout=0.1,
        learnable_adj=True,
        init_gate=-2.0,
        adj_init_off_diag=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.gcn_hidden = gcn_hidden
        self.num_gcn_layers = num_gcn_layers

        # ---------- 可学习邻接矩阵 ----------
        init_adj = torch.eye(num_classes) + adj_init_off_diag * torch.ones(num_classes, num_classes)
        if learnable_adj:
            self.adj = nn.Parameter(init_adj)
        else:
            self.register_buffer("adj", init_adj)

        # ---------- 特征 → 每标签隐状态 ----------
        self.feat_to_label = nn.Sequential(
            nn.Linear(feat_dim, num_classes * gcn_hidden),
            nn.ReLU(inplace=True),
        )

        # ---------- GCN 层 ----------
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(nn.Linear(gcn_hidden, gcn_hidden))
            self.gcn_norms.append(nn.LayerNorm(gcn_hidden))
        self.gcn_dropout = nn.Dropout(dropout)

        # ---------- 每标签隐状态 → 特征空间 ----------
        self.label_to_feat = nn.Sequential(
            nn.Linear(num_classes * gcn_hidden, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # ---------- 门控残差 ----------
        # sigmoid(init_gate) ≈ 0.12，训练初期让原始特征主导，逐步放开
        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, fused_feat):
        """
        参数：
            fused_feat: (B, D)  融合后的特征（如128维）
        返回：
            refined_feat: (B, D)  经标签图增强的特征
        """
        B = fused_feat.shape[0]

        # 1) 投影为每标签隐状态: (B, D) → (B, C, H)
        per_label = self.feat_to_label(fused_feat).view(B, self.num_classes, self.gcn_hidden)

        # 2) 归一化邻接矩阵
        A = F.softmax(self.adj, dim=-1)  # (C, C)  行归一化

        # 3) 多层 GCN: h^{l+1} = σ(A · h^l · W^l) + h^l
        h = per_label
        for gcn_lin, gcn_norm in zip(self.gcn_layers, self.gcn_norms):
            h_prop = torch.matmul(A, h)           # 图传播 (B, C, H)
            h_prop = gcn_lin(h_prop)               # 线性变换
            h_prop = gcn_norm(h_prop)              # LayerNorm 稳定训练
            h_prop = F.relu(h_prop)                # 非线性
            h_prop = self.gcn_dropout(h_prop)
            h = h + h_prop                         # 残差连接

        # 4) 聚合回特征空间: (B, C, H) → (B, C*H) → (B, D)
        aggregated = h.reshape(B, -1)
        aggregated = self.label_to_feat(aggregated)

        # 5) 门控残差注入
        gate = torch.sigmoid(self.gate)
        return fused_feat + gate * aggregated
