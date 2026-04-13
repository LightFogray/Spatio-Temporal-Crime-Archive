"""
时空Transformer架构 - 犯罪预测模型
- 因果时间注意力 (Causal Temporal Attention)
- 空间图注意力 (Spatial Graph Attention)
- 解耦特征融合 (Decoupled Feature Fusion)
- ZINB损失函数
- 可解释性注意力导出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


# ================================
# 1. 数据集类 (复用原有逻辑)
# ================================
class CrimeDataset(Dataset):
    def __init__(self, X, Y, A_crime, OD):
        """
        X: (num_samples, T, N, F) - 动态特征窗口
        Y: (num_samples, N) - 预测目标
        A_crime: (num_samples, N, N) - 动态犯罪图
        OD: (num_samples, N, 4) - OD流特征
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A_crime = torch.tensor(A_crime, dtype=torch.float32)
        self.OD = torch.tensor(OD, dtype=torch.float32)

    def __len__(self):
        return min(len(self.X), len(self.A_crime), len(self.Y))

    def __getitem__(self, idx):
        return (self.X[idx], self.A_crime[idx], self.OD[idx], self.Y[idx])


# ================================
# 2. 位置编码
# ================================
class PositionalEncoding(nn.Module):
    """正弦位置编码 - 用于时间维度"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, H)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        x: (B, T, H) 或 (B, T, N, H)
        """
        if x.dim() == 3:
            # (B, T, H)
            return x + self.pos_embed[:, :x.size(1), :]
        elif x.dim() == 4:
            # (B, T, N, H) - 在时间维度上加位置编码
            B, T, N, H = x.shape
            # pos_embed: (1, T, H) -> (1, T, 1, H) -> broadcast to (B, T, N, H)
            pos = self.pos_embed[:, :T, :].unsqueeze(2)
            return x + pos
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


# ================================
# 3. 因果时间注意力模块
# ================================
class CausalTemporalAttention(nn.Module):
    """
    因果时间自注意力
    - 使用上三角mask保证只看过去
    - 支持多头注意力
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        """
        x: (B, T, H) or (B, T, N, H)
        """
        # 处理4D输入 (B, T, N, H) -> (B*N, T, H)
        if x.dim() == 4:
            B, T, N, H = x.shape
            x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, H)
        else:
            B, T, H = x.shape
            N = None
            x_reshaped = x

        # 因果mask: 上三角为 -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'),
            diagonal=1
        )

        # 自注意力
        residual = x_reshaped
        x_norm = self.norm(x_reshaped)
        x_out, attn_weights = self.attention(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x_out = self.dropout(x_out) + residual

        # 如果输入是4D，reshape回 (B, T, N, H)
        if N is not None:
            x_out = x_out.reshape(B, N, T, H).permute(0, 2, 1, 3)

        if return_attn:
            return x_out, attn_weights
        return x_out


class TemporalTransformerBlock(nn.Module):
    """完整的时间Transformer块"""

    def __init__(self, hidden_dim, num_heads=4, ffn_dim=None, dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # 时间自注意力
        self.attn = CausalTemporalAttention(hidden_dim, num_heads, dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_attn=False):
        # 自注意力
        if return_attn:
            x, attn = self.attn(x, return_attn=True)
        else:
            x = self.attn(x)
            attn = None

        # FFN
        residual = x
        x = self.ffn(self.norm(x)) + residual

        if return_attn:
            return x, attn
        return x


# ================================
# 4. 空间图注意力模块
# ================================
class SpatialGraphAttention(nn.Module):
    """
    空间图注意力层
    - 替代GCN
    - 支持动态图和静态图
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A=None, return_attn=False):
        """
        x: (B, N, H) 或 (B, T, N, H)
        A: (N, N) 或 (B, N, N) - 可选的邻接矩阵先验
        """
        # 处理时间维度
        has_time = x.dim() == 4
        if has_time:
            B, T, N, H = x.shape
            x = x.reshape(B * T, N, H)
        else:
            B, N, H = x.shape
            T = 1

        residual = x
        x = self.norm(x)

        # 多头投影
        q = self.q_proj(x).view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(-1, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B*T, heads, N, head_dim)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*T, heads, N, N)

        # 如果有邻接矩阵先验,作为bias加入
        if A is not None:
            if A.dim() == 2:
                A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            elif A.dim() == 3:
                A = A.unsqueeze(1)  # (B, 1, N, N)
                if has_time:
                    A = A.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B*T, 1, N, N)
            # 将无连接的位置mask掉
            attn = attn.masked_fill(A == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权聚合
        out = torch.matmul(attn, v)  # (B*T, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(-1, N, H)
        out = self.out_proj(out)
        out = self.dropout(out) + residual

        if has_time:
            out = out.reshape(B, T, N, H)

        if return_attn:
            return out, attn
        return out


# ================================
# 5. 可学习自适应图
# ================================
class LearnableGraph(nn.Module):
    """
    可学习的自适应图结构
    - 节点嵌入学习
    - 支持与先验图融合
    """

    def __init__(self, num_nodes, embed_dim=16, prior_weight=0.5):
        super().__init__()
        self.node_embed = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.prior_weight = prior_weight

    def forward(self, A_prior=None):
        """
        返回归一化的邻接矩阵
        """
        # 计算相似度
        A_learned = torch.softmax(self.node_embed @ self.node_embed.T, dim=-1)

        if A_prior is not None:
            # 与先验图融合
            A = self.prior_weight * A_prior + (1 - self.prior_weight) * A_learned
        else:
            A = A_learned

        return A


# ================================
# 6. Cross-Attention 融合模块
# ================================
class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合
    - Query: 动态特征
    - Key/Value: 静态特征
    - 替代简单的门控融合
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, q, kv, return_attn=False):
        """
        q: 动态特征 (B, N, H)
        kv: 静态特征 (B, N, H)
        """
        residual = q

        # Cross-Attention
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv)

        out, attn = self.cross_attn(q_norm, kv_norm, kv_norm)

        # 门控融合
        gate = self.gate(torch.cat([q, out], dim=-1))
        out = gate * out + (1 - gate) * q
        out = out + residual

        # FFN
        out = out + self.ffn(self.norm2(out))

        if return_attn:
            return out, attn
        return out


# ================================
# 7. 超图注意力层
# ================================
class HypergraphAttention(nn.Module):
    """
    超图注意力层
    - 处理POI语义聚类形成的超边
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.spatial_attn = SpatialGraphAttention(hidden_dim, num_heads, dropout)

    def forward(self, x, H_matrix, return_attn=False):
        """
        x: (B, N, H)
        H_matrix: (N, N) - 超图邻接矩阵 (clique expansion)
        """
        # H_matrix 已经是超图的邻接矩阵 (通过 generate_hypergraph_matrix 生成)
        # 直接用于空间注意力
        A_hyper = H_matrix / (H_matrix.sum(dim=-1, keepdim=True) + 1e-6)

        return self.spatial_attn(x, A=A_hyper, return_attn=return_attn)


# ================================
# 8. 自适应专家融合模块 (保守改进方案)
# ================================
class AdaptiveExpertFusion(nn.Module):
    """
    自适应专家融合模块 (Adaptive Expert Fusion)

    核心创新：根据环境复杂度动态选择增强级别
    - 环境复杂度评估：基于功能混合度、异质性等指标
    - 软路由机制：连续权重而非硬划分
    - 三层专家：基础编码 / 语义增强 / CPTED知识约束

    优势：
    1. 避免基于犯罪密度的标签泄漏
    2. 端到端可训练，无需人工阈值
    3. 提供可解释的复杂度分数
    """

    def __init__(self, static_dim, semantic_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ========== 环境复杂度评估器（软路由）==========
        # 计算功能混合度、异质性等复杂度指标
        self.complexity_scorer = nn.Sequential(
            nn.Linear(static_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1的复杂度分数
        )

        # ========== 三层专家 ==========

        # Expert 0: 基础编码器（简单环境）
        self.basic_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Expert 1: 语义增强编码器（中等复杂度）
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Expert 2: CPTED知识增强（高复杂度环境）
        self.cpted_scorer = nn.Sequential(
            nn.Linear(static_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # 4个CPTED维度
            nn.Sigmoid()
        )
        self.cpted_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # ========== 输出投影 ==========
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def compute_cpted_scores(self, x_static):
        """
        计算CPTED四个维度的得分

        x_static包含：POI、道路、照明、绿地、摄像头等

        Returns: (B, N, 4) - [自然监护, 入口控制, 领域强化, 活动支持]
        """
        # 从静态特征中提取相关维度（假设特定索引）
        # 这里简化处理，实际应根据特征定义精确计算

        # 自然监护：照明 + 摄像头 + 可见性
        # 入口控制：道路密度
        # 领域强化：土地利用纯度
        # 活动支持：商业POI密度

        cpted_raw = self.cpted_scorer(x_static)  # (B, N, 4)

        return cpted_raw

    def forward(self, x_static, x_semantic):
        """
        x_static: (B, N, F_s) 原始静态特征
        x_semantic: (N, D) 或 (B, N, D) LLM语义嵌入

        返回:
            - H_out: (B, N, H) 融合后的特征
            - aux_info: dict 包含可解释性信息
        """
        B, N, F_s = x_static.shape

        # 处理语义嵌入维度
        if x_semantic.dim() == 2:
            x_semantic = x_semantic.unsqueeze(0).expand(B, -1, -1)

        # ========== Step 1: 评估环境复杂度 ==========
        complexity = self.complexity_scorer(x_static)  # (B, N, 1)

        # 软路由权重计算
        # 复杂度接近0 -> 主要用Expert 0
        # 复杂度中等 -> 主要用Expert 1
        # 复杂度接近1 -> 主要用Expert 2
        w_basic = (1 - complexity) ** 2  # 低复杂度区域权重高
        w_semantic = 2 * complexity * (1 - complexity)  # 中等复杂度最高
        w_cpted = complexity ** 2  # 高复杂度区域权重高

        # 归一化（确保和为1，数值稳定性）
        w_sum = w_basic + w_semantic + w_cpted + 1e-8
        w_basic = w_basic / w_sum
        w_semantic = w_semantic / w_sum
        w_cpted = w_cpted / w_sum

        # ========== Step 2: 三层专家计算 ==========

        # Expert 0: 基础编码
        h_basic = self.basic_encoder(x_static)  # (B, N, H)

        # Expert 1: 语义增强
        h_static_proj = self.static_proj(x_static)
        h_semantic_proj = self.semantic_proj(x_semantic)
        gate_sem = self.semantic_gate(
            torch.cat([h_static_proj, h_semantic_proj], dim=-1)
        )
        h_semantic = gate_sem * h_semantic_proj + (1 - gate_sem) * h_static_proj

        # Expert 2: CPTED知识增强
        cpted_scores = self.compute_cpted_scores(x_static)  # (B, N, 4)
        h_cpted_input = torch.cat([h_semantic, cpted_scores], dim=-1)
        h_cpted = self.cpted_fusion(h_cpted_input)

        # ========== Step 3: 软路由融合 ==========
        h_fused = (w_basic * h_basic +
                   w_semantic * h_semantic +
                   w_cpted * h_cpted)  # (B, N, H)

        # 输出投影
        H_out = self.output_proj(h_fused)
        H_out = self.norm(H_out)
        H_out = self.dropout(H_out)

        # 可解释性信息
        aux_info = {
            'complexity_score': complexity,  # 环境复杂度 (B, N, 1)
            'expert_weights': torch.cat([w_basic, w_semantic, w_cpted], dim=-1),  # (B, N, 3)
            'semantic_gate': gate_sem,  # 语义门控值
            'cpted_scores': cpted_scores  # CPTED四维度得分
        }

        return H_out, aux_info


# ================================
# 9. 近重复效应建模模块 (创新点)
# ================================
class NearRepeatEffect(nn.Module):
    """
    近重复效应建模模块 (Near-Repeat Effect Modeling)

    理论依据：环境犯罪学中的近重复理论
    - 犯罪事件在空间和时间上呈现聚集性
    - 一次犯罪发生后，附近区域在未来一段时间内发生类似犯罪的概率增加
    - 这种效应随距离和时间衰减

    创新点：
    - 可学习的时空衰减参数，而非固定的高斯核
    - 结合OD流动态调整空间传播
    """

    def __init__(self, num_nodes, hidden_dim):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 可学习的衰减参数
        self.spatial_decay = nn.Parameter(torch.tensor(1.0))   # 空间衰减率 α
        self.temporal_decay = nn.Parameter(torch.tensor(1.0))  # 时间衰减率 β

        # 事件强度编码
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, crime_history, dist_matrix, OD_flow=None):
        """
        近重复效应计算

        crime_history: (B, T, N) 历史犯罪序列
        dist_matrix: (N, N) 距离矩阵
        OD_flow: (B, N, 4) 可选的OD流特征，用于调整传播

        返回: (B, N) 近重复效应强度
        """
        T = crime_history.size(1)
        N = crime_history.size(2)
        device = crime_history.device

        # 1. 时间衰减权重 (越近的事件权重越大)
        time_weights = torch.exp(-self.temporal_decay.abs() * torch.arange(T, device=device).float())
        time_weights = time_weights / (time_weights.sum() + 1e-6)  # 归一化

        # 2. 空间衰减核 (基于距离矩阵)
        # 避免距离为0的情况
        dist_safe = dist_matrix + torch.eye(N, device=device) * 1e6
        spatial_kernel = torch.exp(-self.spatial_decay.abs() * dist_safe)

        # 3. 计算近重复效应
        # 加权历史犯罪
        weighted_history = crime_history * time_weights.view(1, T, 1)  # (B, T, N)

        # 时间聚合
        aggregated_history = weighted_history.sum(dim=1)  # (B, N)

        # 空间传播
        near_repeat_effect = torch.matmul(aggregated_history, spatial_kernel)  # (B, N)

        # 4. 可选：OD流调制 (人流影响犯罪传播)
        if OD_flow is not None:
            # 确保OD_flow在同一设备上
            if isinstance(OD_flow, torch.Tensor):
                OD_flow = OD_flow.to(device)
            od_intensity = OD_flow.mean(dim=-1)  # (B, N)
            modulation = torch.sigmoid(od_intensity)
            near_repeat_effect = near_repeat_effect * (0.5 + 0.5 * modulation)

        # 5. 归一化
        near_repeat_effect = near_repeat_effect / (near_repeat_effect.max() + 1e-6)

        return near_repeat_effect

    def get_decay_params(self):
        """返回学习到的衰减参数，用于可解释性分析"""
        return {
            'spatial_decay': self.spatial_decay.item(),
            'temporal_decay': self.temporal_decay.item()
        }


# ================================
# 10. 完整的时空Transformer模型
# ================================
class SpatioTemporalTransformer(nn.Module):
    """
    完整的时空Transformer架构 (保守改进版)
    - 解耦静态/动态特征处理
    - 因果时间建模
    - 空间图注意力
    - 自适应专家融合 (创新点: 环境复杂度感知的软路由)
    - CPTED知识增强 (创新点: 环境设计知识约束)
    - 近重复效应建模 (创新点: 环境犯罪学理论驱动)
    - 交叉注意力融合
    - ZINB输出
    """

    def __init__(self,
                 static_dim,
                 dynamic_dim,
                 semantic_dim=0,
                 hidden_dim=64,
                 num_heads=4,
                 num_temporal_layers=3,
                 num_spatial_layers=2,
                 dropout=0.1,
                 num_nodes=None,
                 use_semantic_gate=True,      # 是否使用自适应专家融合
                 use_near_repeat=True,        # 是否使用近重复效应
                 distance_matrix=None):       # 距离矩阵 (用于近重复效应)
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_semantic_gate = use_semantic_gate and semantic_dim > 0
        self.use_near_repeat = use_near_repeat
        self.static_dim = static_dim
        self.semantic_dim = semantic_dim

        # ========== 自适应专家融合 (保守改进创新点) ==========
        if self.use_semantic_gate:
            self.adaptive_fusion = AdaptiveExpertFusion(
                static_dim=static_dim,
                semantic_dim=semantic_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            # 静态编码器输入维度
            static_encoder_input = hidden_dim  # 融合输出
        else:
            self.adaptive_fusion = None
            # 无融合时，使用传统拼接
            static_encoder_input = static_dim + semantic_dim

        # ========== 静态特征编码器 ==========
        self.static_encoder = nn.Sequential(
            nn.Linear(static_encoder_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ========== 动态特征编码器 ==========
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 时间位置编码
        self.time_pos = LearnablePositionalEncoding(hidden_dim, max_len=100)

        # ========== 时间Transformer ==========
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_temporal_layers)
        ])

        # ========== 空间注意力层 ==========
        self.spatial_layers = nn.ModuleList([
            SpatialGraphAttention(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_spatial_layers)
        ])

        # ========== 可学习自适应图 ==========
        if num_nodes is not None:
            self.learnable_graph = LearnableGraph(num_nodes, embed_dim=16)
        else:
            self.learnable_graph = None

        # ========== 超图注意力 ==========
        self.hypergraph_attn = HypergraphAttention(hidden_dim, num_heads, dropout)

        # ========== 近重复效应模块 (创新点) ==========
        if self.use_near_repeat and num_nodes is not None:
            self.near_repeat_module = NearRepeatEffect(
                num_nodes=num_nodes,
                hidden_dim=hidden_dim
            )
            # 注册距离矩阵为buffer (如果提供)
            if distance_matrix is not None:
                self.register_buffer('dist_matrix', torch.tensor(distance_matrix, dtype=torch.float32))
            else:
                self.dist_matrix = None

            # 近重复效应融合权重
            self.nr_fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.near_repeat_module = None
            self.dist_matrix = None

        # ========== Cross-Attention融合 ==========
        self.cross_fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)

        # ========== 最终融合层 ==========
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ========== ZINB输出层 ==========
        self.fc_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        self.fc_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, X, A_spatial, A_distance, A_crime, A_hypergraph, OD=None,
                semantic_embed=None, return_attention=False, crime_history=None):
        """
        X: (B, T, N, F) - 输入特征
        A_spatial: (N, N) - 空间邻接矩阵
        A_distance: (N, N) - 距离衰减矩阵
        A_crime: (B, N, N) - 动态犯罪图
        A_hypergraph: (N, N) - 超图邻接矩阵
        OD: (B, N, 4) - OD流特征 (可选)
        semantic_embed: (N, D) - 语义嵌入 (可选)
        crime_history: (B, T, N) - 历史犯罪序列 (用于近重复效应)
        """
        B, T, N, F = X.shape
        adaptive_info = None  # 自适应融合信息
        near_repeat_values = None

        # 分离静态和动态特征
        # 假设前 static_dim 维是静态特征
        static_idx = self.static_dim  # 使用类属性

        X_static = X[:, -1, :, :static_idx]  # 取最后时刻的静态特征 (B, N, F_s)
        X_dynamic = X[:, :, :, static_idx:]   # 动态特征 (B, T, N, F_d)

        # ========== 静态支路 (含自适应专家融合) ==========
        if self.use_semantic_gate and semantic_embed is not None:
            # 保守改进创新点: 自适应专家融合
            H_static_fused, adaptive_info = self.adaptive_fusion(X_static, semantic_embed)
            H_static = self.static_encoder(H_static_fused)
        else:
            # 传统拼接方式
            if semantic_embed is not None:
                if semantic_embed.dim() == 2:
                    # (N, D) -> (B, N, D)
                    semantic_expanded = semantic_embed.unsqueeze(0).expand(B, -1, -1)
                else:
                    # 已经是 (B, N, D)
                    semantic_expanded = semantic_embed
                X_static = torch.cat([X_static, semantic_expanded], dim=-1)
            H_static = self.static_encoder(X_static)

        # 空间传播 (静态)
        H_static = self.spatial_layers[0](H_static, A=A_spatial)
        H_static = self.hypergraph_attn(H_static, A_hypergraph)  # 超图聚合

        # ========== 近重复效应建模 (创新点) ==========
        if self.use_near_repeat and self.near_repeat_module is not None and crime_history is not None:
            # 使用距离矩阵 (优先使用A_distance作为距离信息)
            dist = A_distance if self.dist_matrix is None else self.dist_matrix

            # 计算近重复效应
            near_repeat_effect = self.near_repeat_module(
                crime_history=crime_history,
                dist_matrix=dist,
                OD_flow=OD
            )  # (B, N)

            near_repeat_values = near_repeat_effect

            # 将近重复效应融入特征
            nr_weight = self.nr_fusion_gate(
                torch.cat([H_static, near_repeat_effect.unsqueeze(-1)], dim=-1)
            )  # (B, N, 1)

            H_static = H_static * (1 + nr_weight * near_repeat_effect.unsqueeze(-1))

        # ========== 动态支路 ==========
        H_dynamic = self.dynamic_encoder(X_dynamic)  # (B, T, N, H)
        H_dynamic = self.time_pos(H_dynamic)  # 时间位置编码

        # 时间Transformer (因果)
        temporal_attentions = []
        for layer in self.temporal_layers:
            if return_attention:
                H_dynamic, attn = layer(H_dynamic, return_attn=True)
                temporal_attentions.append(attn)
            else:
                H_dynamic = layer(H_dynamic)

        # 空间注意力 (动态)
        # 对每个时间步应用空间注意力
        H_dynamic_spatial = []
        for t in range(T):
            H_t = self.spatial_layers[1](H_dynamic[:, t, :, :])  # (B, N, H)
            H_dynamic_spatial.append(H_t)
        H_dynamic = torch.stack(H_dynamic_spatial, dim=1)  # (B, T, N, H)

        # 时间聚合 (取最后时刻, 因果性保证)
        H_dynamic_agg = H_dynamic[:, -1, :, :]  # (B, N, H)

        # ========== 融合 ==========
        H_fused = self.cross_fusion(H_dynamic_agg, H_static)  # (B, N, H)

        # 残差连接
        H_final = self.final_norm(H_fused + H_static + H_dynamic_agg)

        # ========== ZINB输出 ==========
        pi = self.fc_pi(H_final).squeeze(-1)      # (B, N)
        mu = self.fc_mu(H_final).squeeze(-1)      # (B, N)
        theta = self.fc_theta(H_final).squeeze(-1) # (B, N)

        if return_attention:
            attention_dict = {
                'temporal_attentions': temporal_attentions,
                'static_feature': H_static,
                'dynamic_feature': H_dynamic_agg,
                'fused_feature': H_final,
                'adaptive_info': adaptive_info,  # 自适应融合信息 (复杂度分数、专家权重)
                'near_repeat_values': near_repeat_values       # 近重复效应强度
            }
            return pi, mu, theta, attention_dict

        return pi, mu, theta, H_static, H_dynamic_agg

    def get_attention_analysis(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                               OD=None, semantic_embed=None, grid_ids=None):
        """
        导出注意力分析结果 (用于可解释性)
        """
        self.eval()
        with torch.no_grad():
            pi, mu, theta, attention_dict = self.forward(
                X, A_spatial, A_distance, A_crime, A_hypergraph, OD,
                semantic_embed, return_attention=True
            )

        analysis = {
            'prediction': {
                'pi': pi.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                'theta': theta.cpu().numpy(),
                'expected_crime': ((1 - pi) * mu).cpu().numpy()
            },
            'attention_weights': {
                'temporal': [attn.cpu().numpy() for attn in attention_dict['temporal_attentions']],
            },
            'feature_contributions': {
                'static_norm': attention_dict['static_feature'].norm(dim=-1).cpu().numpy(),
                'dynamic_norm': attention_dict['dynamic_feature'].norm(dim=-1).cpu().numpy(),
            }
        }

        if grid_ids is not None:
            analysis['grid_ids'] = grid_ids

        return analysis

    def get_feature_importance(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                               OD=None, semantic_embed=None, epsilon=0.01):
        """
        特征重要性分析 (基于输入扰动方法)

        由于模型内部包含近重复效应等可能断开计算图的模块，
        使用基于扰动的方法计算特征重要性，无需反向传播。
        """
        self.eval()

        # 确保输入是tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=A_spatial.device)
        else:
            X = X.to(A_spatial.device)

        with torch.no_grad():
            # 原始预测
            outputs_orig = self.forward(X, A_spatial, A_distance, A_crime,
                                        A_hypergraph, OD, semantic_embed)
            pi_orig, mu_orig = outputs_orig[0], outputs_orig[1]
            pred_orig = ((1 - pi_orig) * mu_orig).sum().item()

            print(f"Computing feature importance for {X.shape[-1]} features...")

            # 对每个特征维度进行扰动
            importance = np.zeros(X.shape[-1])
            for i in range(X.shape[-1]):
                if i % 10 == 0:
                    print(f"  Feature {i}/{X.shape[-1]}...")

                X_perturbed = X.clone()
                X_perturbed[..., i] += epsilon

                outputs_perturbed = self.forward(X_perturbed, A_spatial, A_distance,
                                                 A_crime, A_hypergraph, OD, semantic_embed)
                pi_new, mu_new = outputs_perturbed[0], outputs_perturbed[1]
                pred_new = ((1 - pi_new) * mu_new).sum().item()

                importance[i] = abs(pred_new - pred_orig)

        print("Feature importance computation complete!")
        return importance


# ================================
# 9. ZINB损失函数
# ================================
def zinb_loss(y_true, pi, mu, theta):
    """零膨胀负二项分布损失"""
    eps = 1e-8

    # clamp避免数值爆炸
    pi = torch.clamp(pi, eps, 1 - eps)
    mu = torch.clamp(mu, eps, 1e6)
    theta = torch.clamp(theta, eps, 1e6)

    # log NB likelihood
    t1 = torch.lgamma(theta + y_true)
    t2 = torch.lgamma(theta)
    t3 = torch.lgamma(y_true + 1)

    log_nb = (
        t1 - t2 - t3 +
        theta * (torch.log(theta) - torch.log(theta + mu)) +
        y_true * (torch.log(mu) - torch.log(theta + mu))
    )

    # zero case
    log_zero_nb = theta * (torch.log(theta) - torch.log(theta + mu))

    # log likelihood
    zero_case = torch.log(pi + (1 - pi) * torch.exp(log_zero_nb) + eps)
    non_zero_case = torch.log(1 - pi + eps) + log_nb

    result = torch.where(y_true < 1e-6, zero_case, non_zero_case)

    return -torch.mean(result)


def orthogonality_loss(h_static, h_dynamic):
    """解耦正交约束"""
    B, N, H = h_static.shape
    h_s = h_static.reshape(-1, H)
    h_d = h_dynamic.reshape(-1, H)

    # 正交损失: 最小化两个表示的相关性
    corr = torch.matmul(h_s.T, h_d) / h_s.shape[0]
    loss = torch.norm(corr, p='fro')

    return loss


# ================================
# 10. 训练函数
# ================================
def train_model(model, train_loader, val_loader,
                A_spatial, A_distance, A_hypergraph,
                device, epochs=100, lr=1e-3, weight_decay=1e-5,
                orth_weight=1e-3, clip_norm=5.0, semantic_embed=None,
                crime_history_train=None, crime_history_val=None):
    """
    训练时空Transformer模型

    参数说明:
        semantic_embed: (N, D) LLM增强的语义嵌入
            - 由 generate_semantic_embedding.py 生成
            - 包含POI、路网、土地利用、绿地、天气等环境特征的语义描述
            - 在模型中与静态特征拼接，增强环境感知能力
        crime_history_train/val: 历史犯罪序列，用于近重复效应建模
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # ========== 训练 ==========
        model.train()
        train_losses = []

        for batch_idx, (X_batch, A_crime_batch, OD_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)

            # 处理NaN
            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            optimizer.zero_grad()

            # 提取犯罪历史 (用于近重复效应)
            # 从X中提取犯罪滞后特征 (假设最后7维是犯罪滞后)
            crime_history = X_batch[:, :, :, -7:]  # (B, T, N, 7) -> 取最后一个滞后值
            crime_history = crime_history[:, :, :, 0]  # (B, T, N) 取最近的犯罪值

            # 前向传播 (传入语义嵌入和犯罪历史)
            pi, mu, theta, h_static, h_dynamic = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                semantic_embed=semantic_embed,
                crime_history=crime_history
            )

            # 数值稳定
            mu = torch.clamp(mu, max=100)
            theta = torch.clamp(theta, max=100)

            # 损失计算
            loss_zinb = zinb_loss(Y_batch, pi, mu, theta)
            loss_orth = orthogonality_loss(h_static, h_dynamic)
            loss = loss_zinb + orth_weight * loss_orth

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            train_losses.append(loss.item())

        # ========== 验证 ==========
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, A_crime_batch, OD_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                A_crime_batch = A_crime_batch.to(device)
                OD_batch = OD_batch.to(device)

                if torch.isnan(A_crime_batch).any():
                    A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

                # 提取犯罪历史
                crime_history = X_batch[:, :, :, -7:]
                crime_history = crime_history[:, :, :, 0]

                pi, mu, theta, _, _ = model(
                    X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                    semantic_embed=semantic_embed,
                    crime_history=crime_history
                )

                mu = torch.clamp(mu, max=100)
                theta = torch.clamp(theta, max=100)
                loss = zinb_loss(Y_batch, pi, mu, theta)
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        print(f"Epoch {epoch+1:3d} | Train Loss: {mean_train:.4f} | Val Loss: {mean_val:.4f}")

        # 学习率调度
        scheduler.step()

        # 早停
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "checkpoints/best_model_trans.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 Epoch {epoch+1}")
                break

    return model


# ================================
# 11. 评价指标
# ================================
def calculate_advanced_metrics(y_true, y_pred, k_percent=0.1):
    """计算实战指标"""
    N = y_true.shape[-1]
    k = int(N * k_percent)

    # Top-K Hit Rate
    top_k_pred_idx = np.argsort(y_pred, axis=-1)[:, -k:]
    hits = 0

    for i in range(len(y_true)):
        true_indices = np.where(y_true[i] > 0)[0]
        hits += len(np.intersect1d(top_k_pred_idx[i], true_indices))

    hit_rate = hits / (len(y_true) * k)

    # PAI
    area_ratio = k / N
    captured_crime_ratio = 0
    for i in range(len(y_true)):
        total_sample_crimes = np.sum(y_true[i])
        captured_crimes = np.sum(y_true[i][top_k_pred_idx[i]])
        captured_crime_ratio += (captured_crimes / (total_sample_crimes + 1e-6))

    pai = (captured_crime_ratio / len(y_true)) / area_ratio

    # Jaccard
    top_k_true_idx = np.argsort(y_true, axis=-1)[:, -k:]
    jaccard_sum = 0
    for i in range(len(y_true)):
        intersection = len(np.intersect1d(top_k_pred_idx[i], top_k_true_idx[i]))
        union = len(np.union1d(top_k_pred_idx[i], top_k_true_idx[i]))
        jaccard_sum += intersection / (union + 1e-6)

    jaccard = jaccard_sum / len(y_true)

    return {
        "HitRate": hit_rate,
        "PAI": pai,
        "Jaccard": jaccard
    }


def test_model(model, test_loader, A_spatial, A_distance, A_hypergraph, device, semantic_embed=None):
    """测试模型"""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, A_crime_batch, OD_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)

            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            # 提取犯罪历史
            crime_history = X_batch[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            pi, mu, theta, _, _ = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                semantic_embed=semantic_embed,
                crime_history=crime_history
            )

            pred = torch.clamp((1 - pi) * mu, min=0)
            preds.append(pred.cpu().numpy())
            targets.append(Y_batch.cpu().numpy())

    pred_test = np.vstack(preds)
    Y_test_all = np.vstack(targets)

    # 基础指标
    rmse = np.sqrt(mean_squared_error(Y_test_all.flatten(), pred_test.flatten()))
    mae = mean_absolute_error(Y_test_all.flatten(), pred_test.flatten())

    # 实战指标
    metrics = calculate_advanced_metrics(Y_test_all, pred_test)
    hit_rate = metrics["HitRate"]
    pai = metrics["PAI"]
    jaccard = metrics["Jaccard"]

    print("\n" + "="*50)
    print("测试结果:")
    print("="*50)
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"Hit Rate:  {hit_rate:.4f}")
    print(f"PAI:       {pai:.4f}")
    print(f"Jaccard:   {jaccard:.4f}")
    print("="*50)

    # 打印近重复效应参数 (如果有)
    if hasattr(model, 'near_repeat_module') and model.near_repeat_module is not None:
        decay_params = model.near_repeat_module.get_decay_params()
        print(f"\n近重复效应参数:")
        print(f"  空间衰减率 α: {decay_params['spatial_decay']:.4f}")
        print(f"  时间衰减率 β: {decay_params['temporal_decay']:.4f}")

    return pred_test, Y_test_all


# ================================
# 12. 主程序
# ================================
if __name__ == "__main__":
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== 加载数据 ==========
    print("\n加载数据...")
    X = np.load("data/processed/X.npy")   # (T', N, F)
    Y = np.load("data/processed/Y.npy")   # (T', N)
    OD = np.load("data/processed/dynamic_od_flow_1246.npy")  # (T', N, 4)
    OD = np.log1p(OD)

    A_spatial = np.load("data/processed/adj_adaptive.npy")
    A_distance = np.load("data/processed/adj_distance.npy")
    A_crime_dynamic = np.load("data/processed/adj_crime_dynamic_gaussian.npy")
    A_hypergraph = np.load("data/processed/adj_hypergraph.npy")

    # 加载语义嵌入 (可选)
    # 加载RAG语义嵌入
    semantic_embed_path = "data/processed/semantic_embedding_rag.npy"
    if os.path.exists(semantic_embed_path):
        semantic_embed = np.load(semantic_embed_path)
        print(f"Loaded RAG semantic embedding: {semantic_embed.shape}")
    else:
        # 回退到基础版本
        semantic_embed = np.load("data/processed/semantic_embedding_v2.npy")

    # ========== 构建窗口 ==========
    window = 30
    crime_lag = 7
    offset = window - crime_lag

    X_window = []
    Y_window = []

    for i in range(len(X) - offset):
        X_window.append(X[i:i+offset])
        Y_window.append(Y[i+offset])

    X_window = np.stack(X_window, axis=0).astype(np.float32)
    Y_window = np.stack(Y_window, axis=0).astype(np.float32)

    print(f"X_window: {X_window.shape}")
    print(f"Y_window: {Y_window.shape}")

    # ========== 划分数据集 ==========
    num_samples = X_window.shape[0]
    train_ratio = 0.7
    val_ratio = 0.15

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    X_train, Y_train = X_window[:train_end], Y_window[:train_end]
    X_val, Y_val = X_window[train_end:val_end], Y_window[train_end:val_end]
    X_test, Y_test = X_window[val_end:], Y_window[val_end:]

    A_crime_train = A_crime_dynamic[window:window+train_end]
    A_crime_val = A_crime_dynamic[window+train_end:window+val_end]
    A_crime_test = A_crime_dynamic[window+val_end:window+num_samples]

    OD_train = OD[window:window+train_end]
    OD_val = OD[window+train_end:window+val_end]
    OD_test = OD[window+val_end:window+num_samples]

    # ========== DataLoader ==========
    batch_size = 8

    train_dataset = CrimeDataset(X_train, Y_train, A_crime_train, OD_train)
    val_dataset = CrimeDataset(X_val, Y_val, A_crime_val, OD_val)
    test_dataset = CrimeDataset(X_test, Y_test, A_crime_test, OD_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 模型初始化 ==========
    static_dim = 24
    dynamic_dim = X.shape[2] - static_dim
    semantic_dim = semantic_embed.shape[1] if semantic_embed is not None else 0
    num_nodes = X.shape[1]

    print(f"\n模型参数:")
    print(f"  - static_dim: {static_dim}")
    print(f"  - dynamic_dim: {dynamic_dim}")
    print(f"  - semantic_dim: {semantic_dim}")
    print(f"  - num_nodes: {num_nodes}")

    model = SpatioTemporalTransformer(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        semantic_dim=semantic_dim,
        hidden_dim=64,
        num_heads=4,
        num_temporal_layers=3,
        num_spatial_layers=2,
        dropout=0.1,
        num_nodes=num_nodes,
        use_semantic_gate=True,      # 启用语义门控融合 (创新点)
        use_near_repeat=True,        # 启用近重复效应建模 (创新点)
        distance_matrix=None         # 使用A_distance作为距离信息
    ).to(device)

    print(f"\n创新模块状态:")
    print(f"  - 语义门控融合: {'启用' if model.use_semantic_gate else '禁用'}")
    print(f"  - 近重复效应建模: {'启用' if model.use_near_repeat else '禁用'}")

    # 图矩阵转tensor
    A_spatial = torch.tensor(A_spatial, dtype=torch.float32).to(device)
    A_distance = torch.tensor(A_distance, dtype=torch.float32).to(device)
    A_hypergraph = torch.tensor(A_hypergraph, dtype=torch.float32).to(device)

    if semantic_embed is not None:
        semantic_embed = torch.tensor(semantic_embed, dtype=torch.float32).to(device)

    # ========== 训练 ==========
    print("\n开始训练...")
    model = train_model(
        model, train_loader, val_loader,
        A_spatial, A_distance, A_hypergraph,
        device, epochs=100, lr=1e-3,
        semantic_embed=semantic_embed  # 传入LLM增强的语义嵌入
    )

    # ========== 测试 ==========
    # 加载最佳模型
    model.load_state_dict(torch.load("checkpoints/best_model_trans.pt"))
    pred_test, Y_test_all = test_model(
        model, test_loader, A_spatial, A_distance, A_hypergraph, device,
        semantic_embed=semantic_embed
    )

    # ========== 注意力分析示例 ==========
    print("\n生成注意力分析...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        X_sample = sample_batch[0][:1].to(device)
        A_crime_sample = sample_batch[1][:1].to(device)
        OD_sample = sample_batch[2][:1].to(device)

        analysis = model.get_attention_analysis(
            X_sample, A_spatial, A_distance, A_crime_sample, A_hypergraph,
            OD_sample, semantic_embed
        )

        print(f"预测期望犯罪数: {analysis['prediction']['expected_crime'].shape}")
        print(f"时间注意力层数: {len(analysis['attention_weights']['temporal'])}")

        # 特征重要性
        importance = model.get_feature_importance(
            X_sample, A_spatial, A_distance, A_crime_sample, A_hypergraph,
            OD_sample, semantic_embed
        )
        print(f"特征重要性: {importance.shape}")

    print("\n✅ 训练完成!")
