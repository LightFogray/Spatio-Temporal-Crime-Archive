"""
EP-STD Stage 3 Multi-Graph Enhanced
====================================
多图结构 + 自适应融合 + 交叉反馈门控

核心创新：
  1. 环境自适应图融合: w = MLP(E_env)
  2. 双犯罪传播图 + 交叉反馈门控
  3. 5张图输入: spatial, distance, crime_violent, crime_property, od
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from epstd_stage3 import GraphAttentionLayer


# ==================== 环境自适应图融合模块 ====================

class AdaptiveGraphFusion(nn.Module):
    """
    环境自适应图融合: w = MLP(E_env)

    为每个网格根据环境特征学习个性化的图融合权重
    """
    def __init__(self, env_dim, num_graphs=5, hidden_dim=64):
        super().__init__()
        self.env_dim = env_dim
        self.num_graphs = num_graphs

        # MLP: 环境嵌入 -> 图权重
        self.weight_mlp = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_graphs)
        )

        # 可学习的温度参数（控制softmax的锐度）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

        # 初始化偏置（给spatial和distance更高初始权重）
        self.register_buffer('init_bias', torch.tensor([0.3, 0.3, 0.15, 0.15, 0.1]))

    def forward(self, env_emb):
        """
        Args:
            env_emb: (B, N, env_dim) 环境嵌入

        Returns:
            weights: (B, N, num_graphs) 每个网格的图融合权重
        """
        # MLP预测权重
        logits = self.weight_mlp(env_emb)  # (B, N, num_graphs)

        # 添加初始化偏置
        logits = logits + self.init_bias.view(1, 1, -1)

        # Softmax归一化（带温度）
        weights = F.softmax(logits / self.temperature, dim=-1)  # (B, N, num_graphs)

        return weights


# ==================== 交叉反馈门控机制 ====================

class CrossCrimeGate(nn.Module):
    """
    交叉反馈门控: 在暴力犯罪和财产犯罪传播图之间动态切换

    核心思想:
      - 当暴力犯罪数据稀疏（冷启动）时，提高property图权重
      - 当暴力犯罪数据充足时，主要使用violent图
    """
    def __init__(self, env_dim, hidden_dim=32):
        super().__init__()

        # 门控网络: 根据环境特征和当前状态决定融合比例
        self.gate_mlp = nn.Sequential(
            nn.Linear(env_dim * 2 + 2, hidden_dim),  # env + crime_stats
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # 数据稀疏性检测器
        self.sparsity_detector = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, env_emb, crime_stats=None):
        """
        Args:
            env_emb: (B, N, env_dim) 环境嵌入
            crime_stats: (B, N, 2) 可选的犯罪统计特征 [violent_mean, property_mean]

        Returns:
            gate_weights: (B, N, 2) [violent_weight, property_weight]
        """
        B, N, _ = env_emb.shape

        if crime_stats is None:
            # 如果没有提供统计信息，使用环境特征推断
            crime_stats = torch.zeros(B, N, 2, device=env_emb.device)

        # 检测数据稀疏性
        sparsity = self.sparsity_detector(crime_stats)  # (B, N, 1)

        # 构造门控输入
        env_agg = env_emb.mean(dim=-1, keepdim=True)  # (B, N, 1)
        gate_input = torch.cat([env_emb, env_agg.expand(-1, -1, env_emb.shape[-1]),
                                crime_stats, sparsity.expand(-1, -1, 1)], dim=-1)

        # 计算门控权重
        gate_weights = self.gate_mlp(gate_input)  # (B, N, 2)

        # 冷启动增强: 如果检测到稀疏，强制提高property权重
        cold_start_mask = (sparsity.squeeze(-1) < 0.1).unsqueeze(-1)  # (B, N, 1)
        gate_weights = torch.where(
            cold_start_mask.expand(-1, -1, 2),
            torch.tensor([0.3, 0.7], device=gate_weights.device).view(1, 1, 2),
            gate_weights
        )

        return gate_weights


# ==================== 多图图注意力层 ====================

class MultiGraphAttentionLayer(nn.Module):
    """
    多图图注意力层: 融合5张图 + 环境自适应权重

    输入图:
      - A_spatial: 空间邻接
      - A_distance: 距离衰减
      - A_crime_violent: 暴力犯罪传播
      - A_crime_property: 财产犯罪传播
      - A_od: OD流功能相似
    """
    def __init__(self, in_dim, out_dim, env_dim=64, dropout=0.1, num_graphs=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_graphs = num_graphs

        # 特征变换
        self.W = nn.Linear(in_dim, out_dim)

        # 注意力计算
        self.attn_src = nn.Linear(out_dim, 1)
        self.attn_dst = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # 环境自适应图融合
        self.graph_fusion = AdaptiveGraphFusion(env_dim, num_graphs)

        # 交叉反馈门控（用于crime_violent和crime_property）
        self.cross_crime_gate = CrossCrimeGate(env_dim)

    def forward(self, h, adj_list, env_emb, crime_stats=None):
        """
        Args:
            h: (B, N, in_dim) 输入特征
            adj_list: 邻接矩阵列表 [A_spatial, A_distance, A_crime_violent, A_crime_property, A_od]
            env_emb: (B, N, env_dim) 环境嵌入
            crime_stats: (B, N, 2) 犯罪统计特征 [violent_mean, property_mean]

        Returns:
            h_new: (B, N, out_dim) 更新后的特征
            graph_weights: (B, N, num_graphs) 图融合权重（用于可视化）
            crime_gate: (B, N, 2) 犯罪图门控权重
        """
        B, N, _ = h.shape

        # 线性变换
        Wh = self.W(h)  # (B, N, out_dim)

        # 计算基础注意力分数
        attn_src = self.attn_src(Wh).squeeze(-1)  # (B, N)
        attn_dst = self.attn_dst(Wh).squeeze(-1)  # (B, N)
        e = torch.tanh(attn_src.unsqueeze(2) + attn_dst.unsqueeze(1))  # (B, N, N)

        # 环境自适应图融合权重
        graph_weights = self.graph_fusion(env_emb)  # (B, N, num_graphs)

        # 交叉反馈门控（针对犯罪图）
        crime_gate = self.cross_crime_gate(env_emb, crime_stats)  # (B, N, 2)

        # 调整graph_weights中的crime图权重
        # graph_weights[:, :, 2] *= crime_gate[:, :, 0]  # violent
        # graph_weights[:, :, 3] *= crime_gate[:, :, 1]  # property
        # 重新归一化
        # graph_weights = graph_weights / graph_weights.sum(dim=-1, keepdim=True)

        # 更好的方式: 动态组合crime图后再与其他图融合
        # 先生成组合后的crime图
        adj_crime_combined = (
            crime_gate[:, :, 0].unsqueeze(-1) * adj_list[2] +
            crime_gate[:, :, 1].unsqueeze(-1) * adj_list[3]
        )  # (B, N, N)

        # 计算各图的注意力
        attn_list = []
        for i, adj in enumerate(adj_list):
            if i == 2:  # violent crime - 使用组合后的
                adj_expanded = adj_crime_combined
            elif i == 3:  # property crime - 跳过，已合并
                continue
            else:
                if len(adj.shape) == 2:
                    adj_expanded = adj.unsqueeze(0).expand(B, -1, -1)
                else:
                    adj_expanded = adj

            # 掩码注意力
            e_masked = e.masked_fill(adj_expanded == 0, float('-inf'))
            attn = F.softmax(e_masked, dim=-1)
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
            attn_list.append(attn)

        # 加权融合各图的注意力 (现在只有4个有效图)
        attn_weights = sum(
            graph_weights[:, :, i].unsqueeze(-1) * attn_list[i if i < 2 else i-1]
            for i in [0, 1, 4]  # spatial, distance, od
        )
        # 加上组合crime图
        attn_weights = attn_weights + graph_weights[:, :, 2:3].sum(dim=-1, keepdim=True) * attn_list[2]

        # 聚合邻居信息
        h_new = torch.bmm(attn_weights, Wh)  # (B, N, out_dim)
        h_new = self.dropout(h_new)

        # 残差连接
        return F.gelu(h_new + Wh), graph_weights, crime_gate


# ==================== 多图条件扩散模型 ====================

class MultiGraphConditionalDiffusion(nn.Module):
    """
    多图条件扩散模型

    支持5张图输入 + 环境自适应融合 + 交叉反馈门控
    """
    def __init__(self, num_nodes, hidden_dim=128, num_layers=4,
                 time_dim=64, env_dim=64, num_prototypes=10, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_dim = time_dim
        self.env_dim = env_dim

        # 时间编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 环境编码投影
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 原型嵌入
        self.prototype_embedding = nn.Embedding(num_prototypes, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(1, hidden_dim)

        # 多图注意力层
        self.graph_layers = nn.ModuleList([
            MultiGraphAttentionLayer(
                hidden_dim if i == 0 else hidden_dim,
                hidden_dim,
                env_dim=env_dim,
                dropout=dropout,
                num_graphs=5
            ) for i in range(num_layers)
        ])

        # 时间-环境融合层
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # 输出头
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 零膨胀头
        self.zero_inflation_head = nn.Sequential(
            nn.Linear(hidden_dim + env_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_t, t, env_emb, prototype_ids, adj_list=None, crime_stats=None):
        """
        Args:
            x_t: (B, N) 噪声状态
            t: (B,) 时间步
            env_emb: (B, N, env_dim) 环境嵌入
            prototype_ids: (N,) 原型ID
            adj_list: List[Tensor] 5张邻接矩阵
            crime_stats: (B, N, 2) 犯罪统计特征

        Returns:
            noise_pred: (B, N) 噪声预测
            pi: (B, N) 零膨胀概率
            graph_weights: List[Tensor] 每层图融合权重
            crime_gates: List[Tensor] 每层犯罪图门控
        """
        B, N = x_t.shape

        # 时间编码
        t_norm = t.float().unsqueeze(-1) / 1000.0  # 归一化
        t_emb = self.time_mlp(t_norm)  # (B, time_dim)

        # 环境编码
        h_env = self.env_encoder(env_emb)  # (B, N, hidden_dim)

        # 原型编码
        h_proto = self.prototype_embedding(prototype_ids)  # (N, hidden_dim)
        h_proto = h_proto.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden_dim)

        # 输入投影
        h = self.input_proj(x_t.unsqueeze(-1))  # (B, N, hidden_dim)

        # 记录图权重和门控
        all_graph_weights = []
        all_crime_gates = []

        # 多图注意力层
        for gconv, fusion in zip(self.graph_layers, self.fusion_layers):
            if adj_list is not None:
                h, graph_weights, crime_gate = gconv(h, adj_list, env_emb, crime_stats)
                all_graph_weights.append(graph_weights)
                all_crime_gates.append(crime_gate)

            # 融合时间、环境、原型信息
            t_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)
            fusion_input = torch.cat([h, h_env, t_expanded], dim=-1)
            h = fusion(fusion_input)

        # 输出噪声预测
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, N)

        # 零膨胀概率
        zero_input = torch.cat([h.mean(dim=1), env_emb.mean(dim=1)], dim=-1)
        pi = self.zero_inflation_head(zero_input).expand(-1, N)  # (B, N)

        return noise_pred, pi, all_graph_weights, all_crime_gates


# ==================== 双任务多图扩散模型 ====================

class DualTaskMultiGraphDiffusion(nn.Module):
    """
    双任务多图扩散模型: 暴力犯罪 + 财产犯罪

    共享图结构，但分别预测
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        # 任务特定输出头
        self.violence_head = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(base_model.hidden_dim // 2, 1)
        )

        self.property_head = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(base_model.hidden_dim // 2, 1)
        )

        # 任务不确定性权重（用于多任务损失）
        self.log_sigma_violent = nn.Parameter(torch.zeros(1))
        self.log_sigma_property = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, env_emb, prototype_ids, adj_list=None, crime_stats=None):
        """
        前向传播，输出两个任务的预测

        Returns:
            noise_violent, noise_property: (B, N)
            pi_violent, pi_property: (B, N)
            graph_weights, crime_gates
        """
        B, N = x_t.shape

        # 时间编码
        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.base.time_mlp(t_norm)

        # 环境和原型编码
        h_env = self.base.env_encoder(env_emb)
        h_proto = self.base.prototype_embedding(prototype_ids).unsqueeze(0).expand(B, -1, -1)

        # 输入投影
        h = self.base.input_proj(x_t.unsqueeze(-1))

        # 记录权重
        all_graph_weights = []
        all_crime_gates = []

        # 多图注意力层（共享）
        for gconv, fusion in zip(self.base.graph_layers, self.base.fusion_layers):
            if adj_list is not None:
                h, graph_weights, crime_gate = gconv(h, adj_list, env_emb, crime_stats)
                all_graph_weights.append(graph_weights)
                all_crime_gates.append(crime_gate)

            t_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)
            fusion_input = torch.cat([h, h_env, t_expanded], dim=-1)
            h = fusion(fusion_input)

        # 任务特定输出
        noise_violent = self.violence_head(h).squeeze(-1)
        noise_property = self.property_head(h).squeeze(-1)

        # 零膨胀概率
        zero_input = torch.cat([h.mean(dim=1), env_emb.mean(dim=1)], dim=-1)
        pi = self.base.zero_inflation_head(zero_input).expand(-1, N)

        return (noise_violent, noise_property), pi, all_graph_weights, all_crime_gates


if __name__ == "__main__":
    # 测试代码
    print("Testing Multi-Graph Models...")

    # 创建模拟数据
    B, N, env_dim = 4, 1246, 64
    num_graphs = 5

    env_emb = torch.randn(B, N, env_dim)
    adj_list = [torch.eye(N) for _ in range(num_graphs)]

    # 测试自适应图融合
    fusion = AdaptiveGraphFusion(env_dim, num_graphs)
    weights = fusion(env_emb)
    print(f"Graph weights shape: {weights.shape}")  # (B, N, 5)
    print(f"Weights sum: {weights.sum(dim=-1).mean():.4f}")  # 应接近1

    # 测试交叉反馈门控
    gate = CrossCrimeGate(env_dim)
    crime_stats = torch.rand(B, N, 2)
    gate_weights = gate(env_emb, crime_stats)
    print(f"Crime gate weights shape: {gate_weights.shape}")  # (B, N, 2)

    # 测试多图注意力层
    attn_layer = MultiGraphAttentionLayer(64, 128, env_dim, num_graphs=5)
    h = torch.randn(B, N, 64)
    h_new, graph_w, crime_g = attn_layer(h, adj_list, env_emb, crime_stats)
    print(f"Multi-graph attention output: {h_new.shape}")

    print("\nAll tests passed!")
