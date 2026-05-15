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
import sys
import yaml
from types import SimpleNamespace

"""
加载模型训练超参数配置
支持递归访问和动态覆盖
cfg = load_config("src/config/var_conf.yml", overrides={"training.batch_size": 16, "training.learning_rate": 5e-4})
"""
def load_config(config_path="src/config/var_conf.yml", overrides=None):
    """
    加载 YAML 配置文件，并允许动态覆盖部分参数
    
    参数:
        config_path: str - 配置文件路径
        overrides: dict - 可选的参数覆盖字典，例如 {"training.batch_size": 16, "training.learning_rate": 0.0005}
    
    返回:
        cfg: SimpleNamespace - 可通过 cfg.dataset.batch_size 访问
    """
    with open(config_path, "r", encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    # 递归转换 dict -> SimpleNamespace
    def dict2namespace(d):
        ns = SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict2namespace(v))
            else:
                setattr(ns, k, v)
        return ns

    cfg = dict2namespace(cfg_dict)

    # 应用覆盖参数
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            target = cfg
            for k in keys[:-1]:
                target = getattr(target, k)
            setattr(target, keys[-1], value)

    return cfg
cfg = load_config("src/config/var_conf.yml")

# ================================
# 1. 数据集类 (复用原有逻辑)
# ================================
class CrimeDataset(Dataset):
    def __init__(self, X, Y, A_crime, OD, A_property=None, Y_property=None):
        """
        X: (num_samples, T, N, F) - 动态特征窗口
        Y: (num_samples, N) - 预测目标（暴力犯罪）
        A_crime: (num_samples, N, N) - 动态犯罪图（暴力）
        OD: (num_samples, N, 4) - OD流特征
        A_property: (num_samples, N, N) - 动态犯罪图（财产，可选）
        Y_property: (num_samples, N) - 财产犯罪标签（双任务学习）
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A_crime = torch.tensor(A_crime, dtype=torch.float32)
        self.OD = torch.tensor(OD, dtype=torch.float32)
        if A_property is not None:
            self.A_property = torch.tensor(A_property, dtype=torch.float32)
        else:
            self.A_property = None
        # 财产犯罪标签（用于双任务学习）
        if Y_property is not None:
            self.Y_property = torch.tensor(Y_property, dtype=torch.float32)
        else:
            self.Y_property = None

    def __len__(self):
        lengths = [len(self.X), len(self.A_crime), len(self.OD), len(self.Y)]
        if self.A_property is not None:
            lengths.append(len(self.A_property))
        if self.Y_property is not None:
            lengths.append(len(self.Y_property))
        return min(lengths)

    def __getitem__(self, idx):
        items = [self.X[idx], self.A_crime[idx], self.OD[idx], self.Y[idx]]
        if self.Y_property is not None:
            items.append(self.Y_property[idx])
        elif self.A_property is not None:
            items.append(self.A_property[idx])
        return tuple(items)


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
    可学习的自适应图结构 - Top-k稀疏化版本
    - 节点嵌入学习
    - Top-k稀疏化防止过平滑
    - 支持与先验图融合
    """

    def __init__(self, num_nodes, embed_dim=16, prior_weight=0.5, k=10):
        super().__init__()
        self.node_embed = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.prior_weight = prior_weight
        self.k = k  # Top-k邻居数

    def forward(self, A_prior=None):
        """
        返回归一化的邻接矩阵
        """
        # 计算相似度
        similarity = self.node_embed @ self.node_embed.T  # (N, N)

        # Top-k稀疏化: 只保留每个节点最相似的k个邻居
        N = similarity.size(0)
        # 对角线设为负无穷，避免选自己
        similarity_no_diag = similarity.clone()
        similarity_no_diag.fill_diagonal_(-float('inf'))

        # 获取Top-k索引
        _, topk_indices = torch.topk(similarity_no_diag, k=min(self.k, N-1), dim=-1)  # (N, k)

        # 构建稀疏邻接矩阵
        A_learned = torch.zeros_like(similarity)
        row_indices = torch.arange(N, device=similarity.device).unsqueeze(1).expand(-1, topk_indices.size(1))
        A_learned.scatter_(1, topk_indices, 1.0)

        # 对称化 (如果i连接j，则j也连接i)
        A_learned = torch.max(A_learned, A_learned.T)

        # 归一化
        degree = A_learned.sum(dim=-1, keepdim=True) + 1e-8
        A_learned = A_learned / degree

        if A_prior is not None:
            # 与先验图融合
            A = self.prior_weight * A_prior + (1 - self.prior_weight) * A_learned
            # 重新归一化
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
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
# 9. 门控融合
# ================================
class GatingFusion(nn.Module):
    """
    门控竞争融合
    
    核心设计：
    - 使用门控机制让历史特征(h_basic)和KG特征(h_kg)竞争
    - 而非简单的加性融合或软路由
    - 竞争公式：h = g * h_basic + (1-g) * h_kg
    
    优势：
    - 保持特征局部显著性，缓解过平滑
    - 空间不同位置可动态选择特征源
    - 提升Jaccard指标
    """

    def __init__(self, static_dim, hidden_dim, dropout=0.1, kg_dim=32,
                 crime_history_dim=7, density_threshold=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kg_dim = kg_dim
        self.crime_history_dim = crime_history_dim  # 历史犯罪特征维度
        # density_threshold: 划分阈值，None则自动用批次中位数
        self.density_threshold = density_threshold
        # 运行统计量用于归一化
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_std', torch.tensor(1.0))
        self.register_buffer('num_batches', torch.tensor(0))

        # KG投影层
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 基础编码器（历史特征）
        self.basic_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 门控竞争融合：计算历史特征 vs KG特征的权重
        self.high_freq_scale = nn.Parameter(torch.tensor(1.0))  # g=1时全用历史特征，g=0时全用KG特征

        # 门控竞争融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def compute_crime_density(self, crime_history):
        """
        计算犯罪密度分数
        crime_history: (B, N, T_crime) 历史犯罪序列
        Returns: (B, N, 1) 密度分数 0-1
        """
        # 均值作为密度指标
        mean_crime = crime_history.mean(dim=-1, keepdim=True)
        # 使用全局统计量归一化（而不是批次中位数）
        #  crimes通常服从幂律分布，取log后归一化
        log_density = torch.log1p(mean_crime)
        # 使用固定的全局阈值（约0.5对应原始值0.65左右）
        # 或者用运行均值估计
        if self.training:
            # 训练时使用运行统计量
            with torch.no_grad():
                batch_mean = log_density.mean()
                batch_std = log_density.std() + 1e-6
                # 更新运行统计量
                self.running_mean = 0.9 * self.running_mean + 0.1 * batch_mean
                self.running_std = 0.9 * self.running_std + 0.1 * batch_std

        # Z-score标准化到0-1
        normalized = (log_density - self.running_mean) / (self.running_std + 1e-6)
        # 映射到0-1，0.5为阈值
        density = torch.sigmoid(normalized)
        return density

    def forward(self, x_static, kg_embed, crime_history=None):
        """
        x_static: (B, N, F_s)
        kg_embed: (N, D_kg) 或 (B, N, D_kg)
        crime_history: (B, N, T) 历史犯罪序列，用于密度计算
        """
        B, N, F_s = x_static.shape

        # 处理KG维度
        if kg_embed.dim() == 2:
            kg_embed = kg_embed.unsqueeze(0).expand(B, -1, -1)
        h_kg = self.kg_proj(kg_embed)

        # 历史特征编码
        h_basic = self.basic_encoder(x_static)

        # 计算犯罪密度（如果没有提供crime_history，则假设全低频）
        if crime_history is not None:
            crime_density = self.compute_crime_density(crime_history)  # (B, N, 1)
        else:
            crime_density = torch.zeros(B, N, 1, device=x_static.device)

        # 门控竞争融合：让历史特征和KG特征竞争
        # 计算竞争门控
        gate_input = torch.cat([h_basic, h_kg], dim=-1)
        g = self.fusion_gate(gate_input)  # (B, N, 1)
        
        # 竞争融合：g * h_basic + (1-g) * h_kg
        # 这种设计强制模型在空间不同位置选择更重要的特征源
        H_fused = g * h_basic + (1 - g) * h_kg

        # 输出投影
        H_out = self.output_proj(H_fused)
        H_out = self.norm(H_out)
        H_out = self.dropout(H_out)

        # 可解释性信息
        gate_mean = g.mean().item()
        aux_info = {
            'expert_weights': g,  # (B, N, 1) 门控权重，越大越依赖历史特征
            'complexity_score': crime_density,  # 密度分数
            'kg_enhanced': True,
            'history_ratio': gate_mean,  # 依赖历史特征的比例
        }

        return H_out, aux_info
# ================================
# 9. 近重复效应建模模块 (创新点)
# ================================
# ================================
# 多类型犯罪传播模块
# ================================
class MultiTypeNearRepeat(nn.Module):
    """
    多类型近重复效应模块 - 共享衰减参数，独立类型调制

    设计原则:
    - 共享衰减参数 α/β（防止过拟合）
    - 独立的历史强度编码器（捕捉类型特定模式）
    - 类型特定的输出门控（差异化融合权重）
    """

    CRIME_TYPES = ['violent', 'property', 'public']

    def __init__(self, num_nodes, hidden_dim, num_types=3):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_types = num_types

        # 共享的衰减参数（所有类型共享，防止过拟合）
        self.spatial_decay = nn.Parameter(torch.tensor(1.0))   # α
        self.temporal_decay = nn.Parameter(torch.tensor(1.0))  # β

        # 类型特定的衰减调制（微调参数）
        self.decay_modulation = nn.ParameterDict({
            ctype: nn.Parameter(torch.tensor([0.0, 0.0]))  # [α_mod, β_mod]
            for ctype in self.CRIME_TYPES[:num_types]
        })

        # 共享的历史强度编码器（FFN共用）
        self.shared_intensity_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 类型特定的输出门控（差异化融合）
        self.type_gates = nn.ModuleDict({
            ctype: nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            for ctype in self.CRIME_TYPES[:num_types]
        })

    def get_effective_decay(self, crime_type):
        """获取类型有效的衰减参数"""
        base_alpha = self.spatial_decay.abs()
        base_beta = self.temporal_decay.abs()

        modulation = self.decay_modulation[crime_type]

        # 基础参数 + 类型特定微调
        alpha = base_alpha * (1 + modulation[0])
        beta = base_beta * (1 + modulation[1])

        return alpha, beta

    def forward(self, crime_history, dist_matrix, crime_type='violent', OD_flow=None):
        """
        计算特定类型的近重复效应

        crime_history: (B, T, N) 该类型犯罪的历史序列
        dist_matrix: (N, N) 距离矩阵
        crime_type: 犯罪类型 ('violent', 'property', 'public')
        OD_flow: (B, N, 4) OD流特征

        返回: (B, N) 近重复效应强度
        """
        B, T, N = crime_history.shape
        device = crime_history.device

        # 获取类型有效的衰减参数
        alpha, beta = self.get_effective_decay(crime_type)

        # 1. 时间衰减权重
        time_weights = torch.exp(-beta * torch.arange(T, device=device).float())
        time_weights = time_weights / (time_weights.sum() + 1e-6)

        # 2. 空间衰减核
        dist_safe = dist_matrix + torch.eye(N, device=device) * 1e6
        spatial_kernel = torch.exp(-alpha * dist_safe)

        # 3. 历史强度编码（共享FFN）
        history_flat = crime_history.reshape(-1, 1)
        encoded_intensity = self.shared_intensity_encoder(history_flat)
        encoded_history = encoded_intensity.reshape(B, T, N)

        # 4. 计算近重复效应
        weighted_history = encoded_history * time_weights.view(1, T, 1)
        aggregated_history = weighted_history.sum(dim=1)  # (B, N)
        near_repeat_effect = torch.matmul(aggregated_history, spatial_kernel)  # (B, N)

        # 5. OD流调制
        if OD_flow is not None:
            od_intensity = OD_flow.mean(dim=-1)  # (B, N)
            modulation = torch.sigmoid(od_intensity)
            near_repeat_effect = near_repeat_effect * (0.5 + 0.5 * modulation)

        # 6. 类型特定门控输出
        effect_normalized = near_repeat_effect / (near_repeat_effect.max() + 1e-6)
        gate_weight = self.type_gates[crime_type](effect_normalized.unsqueeze(-1)).squeeze(-1)

        return effect_normalized * gate_weight

    def get_all_type_effects(self, crime_histories, dist_matrix, OD_flow=None):
        """
        一次性计算所有犯罪类型的近重复效应

        crime_histories: dict {type: (B, T, N)}
        返回: dict {type: (B, N)}
        """
        effects = {}
        for ctype in self.CRIME_TYPES[:self.num_types]:
            if ctype in crime_histories and crime_histories[ctype] is not None:
                effects[ctype] = self.forward(
                    crime_histories[ctype], dist_matrix, ctype, OD_flow
                )
            else:
                effects[ctype] = None
        return effects

    def get_decay_params(self):
        """返回所有类型的有效衰减参数"""
        params = {}
        for ctype in self.CRIME_TYPES[:self.num_types]:
            alpha, beta = self.get_effective_decay(ctype)
            params[ctype] = {
                'spatial_decay': alpha.item(),
                'temporal_decay': beta.item()
            }
        return params


class CrossTypeGating(nn.Module):
    """
    交叉类型门控融合模块 - 解决信息传递和冷启动
    """

    def __init__(self, hidden_dim, kg_dim=0, num_types=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.use_kg = kg_dim > 0

        # 数据可用性编码
        self.availability_encoder = nn.Sequential(
            nn.Linear(num_types, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # KG增强
        if self.use_kg:
            self.kg_proj = nn.Sequential(
                nn.Linear(kg_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU()
            )
            cross_input_dim = hidden_dim * num_types + hidden_dim // 2 + hidden_dim // 2
        else:
            cross_input_dim = hidden_dim * num_types + hidden_dim // 2

        # 门控网络
        self.fusion_gate = nn.Sequential(
            nn.Linear(cross_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_types),
            nn.Softmax(dim=-1)
        )

        # 类型间交互Transformer
        self.cross_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, type_features, availability_mask, kg_embed=None):
        """
        交叉类型融合

        type_features: dict {type: (B, N, H)} 各类型特征
        availability_mask: (B, N, num_types) 数据可用性 [0, 1]
        kg_embed: (B, N, kg_dim) KG嵌入

        返回: (B, N, H) 融合特征, (B, N, num_types) 融合权重
        """
        B, N, _ = availability_mask.shape
        device = availability_mask.device

        # 收集所有类型特征
        type_list = ['violent', 'property', 'public'][:self.num_types]
        features_stack = []

        for ctype in type_list:
            if ctype in type_features and type_features[ctype] is not None:
                features_stack.append(type_features[ctype])
            else:
                features_stack.append(torch.zeros(B, N, self.hidden_dim, device=device))

        # (B, N, num_types*H)
        concat_features = torch.cat(features_stack, dim=-1)

        # 编码数据可用性
        avail_encoding = self.availability_encoder(availability_mask)

        # 构建门控输入
        if self.use_kg and kg_embed is not None:
            kg_encoded = self.kg_proj(kg_embed)
            gate_input = torch.cat([concat_features, avail_encoding, kg_encoded], dim=-1)
        else:
            gate_input = torch.cat([concat_features, avail_encoding], dim=-1)

        # 计算融合权重
        fusion_weights = self.fusion_gate(gate_input)

        # 根据可用性调整权重
        adjusted_weights = fusion_weights * availability_mask
        adjusted_weights = adjusted_weights / (adjusted_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 类型间交互
        stacked = torch.stack(features_stack, dim=-2)  # (B, N, num_types, H)
        weighted = stacked * adjusted_weights.unsqueeze(-1)

        # Cross-Transformer
        transformer_input = weighted.reshape(B * N, self.num_types, self.hidden_dim)
        transformed = self.cross_transformer(transformer_input)
        transformed = transformed.reshape(B, N, self.num_types, self.hidden_dim)

        # 聚合（加入残差连接，保留violent特征，避免过度平滑）
        aggregated = transformed.sum(dim=2) + features_stack[0]  # + violent残差

        return aggregated, adjusted_weights

# ================================
# 10. 完整的时空Transformer模型
# ================================
class SpatioTemporalTransformer(nn.Module):
    """
    完整的时空Transformer架构 (保守改进版)
    - 解耦静态/动态特征处理
    - 因果时间建模
    - 空间图注意力
    - CPTED知识增强 (创新点: 环境设计知识约束)
    - 近重复效应建模 (创新点: 环境犯罪学理论驱动)
    - 交叉注意力融合
    - ZINB输出
    """

    def __init__(self,
                 static_dim,
                 dynamic_dim,
                 kg_dim=0,                    # 知识图谱嵌入维度
                 hidden_dim=64,
                 num_heads=4,
                 num_temporal_layers=3,
                 num_spatial_layers=2,
                 dropout=0.1,
                 num_nodes=None,
                 use_multitype_nr=True,       # 使用多类型近重复效应
                 distance_matrix=None,
                 predict_property=False,      # 是否同时预测财产犯罪（双任务）
                 predict_public=False):       # 是否预测公共秩序犯罪（三任务）
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_multitype_nr = use_multitype_nr
        self.static_dim = static_dim

        # 确定预测类型
        self.predict_types = ['violent']
        if predict_property:
            self.predict_types.append('property')
        if predict_public:
            self.predict_types.append('public')
        self.num_predict_types = len(self.predict_types)

        # ========== 门控竞争融合 (保守改进创新点) ==========
        self.gating_fusion = GatingFusion(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kg_dim=kg_dim  # 传入KG维度
        )
        # 静态编码器输入维度
        # adaptive_fusion输出的是融合后的特征，维度为hidden_dim（经过output_proj）
        static_encoder_input = hidden_dim

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

        # ========== 多类型近重复效应模块 (改进版) ==========
        if self.use_multitype_nr and num_nodes is not None:
            self.multitype_nr = MultiTypeNearRepeat(
                num_nodes=num_nodes,
                hidden_dim=hidden_dim,
                num_types=3  # 三种犯罪类型
            )
            # 注册距离矩阵为buffer
            if distance_matrix is not None:
                self.register_buffer('dist_matrix', torch.tensor(distance_matrix, dtype=torch.float32))
            else:
                self.dist_matrix = None
            # Near Repeat门控机制：控制NR特征强度，防止高频区过强
            self.nr_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.multitype_nr = None
            self.dist_matrix = None
            self.nr_gate = None

        # ========== 交叉类型门控融合 (解决信息传递) ==========
        if self.num_predict_types > 1:
            self.cross_type_gating = CrossTypeGating(
                hidden_dim=hidden_dim,
                kg_dim=kg_dim,
                num_types=self.num_predict_types
            )
        else:
            self.cross_type_gating = None

        # ========== Cross-Attention融合 ==========
        self.cross_fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)

        # ========== 最终融合层 ==========
        self.final_norm = nn.LayerNorm(hidden_dim)

        # ========== 类型特定ZINB输出层 ==========
        self.type_outputs = nn.ModuleDict()
        for ctype in self.predict_types:
            self.type_outputs[ctype] = nn.ModuleDict({
                'pi': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                ),
                'mu': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Softplus()
                ),
                'theta': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Softplus()
                )
            })

        # 保留向后兼容的属性
        self.predict_property = predict_property

    def forward(self, X, A_spatial, A_distance, A_crime, A_hypergraph, OD=None,
                kg_embed=None, return_attention=False, crime_history=None,
                crime_history_dict=None, return_adaptive_info=False):
        """
        X: (B, T, N, F) - 输入特征
        A_spatial: (N, N) - 空间邻接矩阵
        A_distance: (N, N) - 距离衰减矩阵
        A_crime: (B, N, N) - 动态犯罪图 (暴力犯罪)
        A_hypergraph: (N, N) - 超图邻接矩阵
        OD: (B, N, 4) - OD流特征 (可选)
        crime_history: (B, T, N) - 历史犯罪序列 (向后兼容，优先使用crime_history_dict)
        crime_history_dict: dict {type: (B, T, N)} - 各类型历史犯罪序列
        """
        B, T, N, F = X.shape
        device = X.device
        adaptive_info = None
        fusion_weights = None
        cold_info_dict = {}

        # 处理犯罪历史字典（向后兼容）
        if crime_history_dict is None and crime_history is not None:
            # 如果提供了单一的crime_history，转换为dict格式
            crime_history_dict = {'violent': crime_history}
        elif crime_history_dict is None:
            crime_history_dict = {}

        # 分离静态和动态特征
        static_idx = self.static_dim
        X_static = X[:, -1, :, :static_idx]  # (B, N, F_s)
        X_dynamic = X[:, :, :, static_idx:]   # (B, T, N, F_d)

        # 从主要犯罪历史提取静态密度特征
        primary_type = self.predict_types[0]
        if primary_type in crime_history_dict and crime_history_dict[primary_type] is not None:
            static_crime_history = crime_history_dict[primary_type].transpose(1, 2)  # (B, N, T)
        else:
            static_crime_history = None

        # ========== 静态支路 (含门控竞争融合) ==========
        H_static_fused, adaptive_info = self.gating_fusion(X_static, kg_embed, static_crime_history)
        H_static = self.static_encoder(H_static_fused)

        # 空间传播 (静态)
        H_static = self.spatial_layers[0](H_static, A=A_spatial)
        H_static = self.hypergraph_attn(H_static, A_hypergraph)

        # ========== 多类型近重复效应建模 (改进版) ==========
        nr_features = {}
        if self.use_multitype_nr and self.multitype_nr is not None and crime_history_dict:
            dist = A_distance if self.dist_matrix is None else self.dist_matrix

            # 计算所有类型的近重复效应
            nr_effects = self.multitype_nr.get_all_type_effects(
                crime_history_dict, dist, OD
            )

            # 构建各类型NR特征
            for ctype in self.predict_types:
                if nr_effects.get(ctype) is not None:
                    nr_features[ctype] = nr_effects[ctype].unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                else:
                    nr_features[ctype] = torch.zeros(B, N, self.hidden_dim, device=device)

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
        H_dynamic_agg = self.temporal_pooling(H_dynamic)  # (B, N, H) - Attention Pooling

        # ========== 类型特定特征准备 ==========
        type_features = {}
        availability_mask = []

        for ctype in self.predict_types:
            # 合并静态、动态、近重复特征
            if ctype in nr_features:
                nr_gate = torch.sigmoid(self.nr_gate(H_dynamic_agg))

                combined = (
                    H_static
                    + H_dynamic_agg
                    + nr_gate * nr_features[ctype]
                )
            else:
                combined = H_static + H_dynamic_agg

            # Cross-Attention融合
            type_features[ctype] = self.cross_fusion(H_dynamic_agg, combined)

            # 检查数据可用性（基于历史犯罪是否为零）
            if ctype in crime_history_dict and crime_history_dict[ctype] is not None:
                has_data = (crime_history_dict[ctype].sum(dim=(1, 2)) > 0).float().unsqueeze(1)
                availability_mask.append(has_data.expand(B, N))
            else:
                availability_mask.append(torch.ones(B, N, device=device))

        availability_tensor = torch.stack(availability_mask, dim=-1)  # (B, N, num_types)

        # ========== 交叉类型门控融合 ==========
        if self.cross_type_gating is not None and len(self.predict_types) > 1:
            fused_features, fusion_weights = self.cross_type_gating(
                type_features, availability_tensor, kg_embed
            )
        else:
            # 单类型预测，直接使用类型特征
            fused_features = type_features[self.predict_types[0]]
            fusion_weights = availability_tensor

        # 残差连接
        H_final = self.final_norm(fused_features + H_static + H_dynamic_agg)

        # ========== 类型特定ZINB输出 ==========
        outputs = {}
        for ctype in self.predict_types:
            pi = self.type_outputs[ctype]['pi'](H_final).squeeze(-1)
            mu = self.type_outputs[ctype]['mu'](H_final).squeeze(-1)
            theta = self.type_outputs[ctype]['theta'](H_final).squeeze(-1)
            outputs[ctype] = (pi, mu, theta)

        # 向后兼容：提取暴力犯罪和财产犯罪预测
        pi, mu, theta = outputs[self.predict_types[0]]
        if 'property' in outputs:
            pi_property, mu_property, theta_property = outputs['property']
        else:
            pi_property = mu_property = theta_property = None

        if return_attention:
            attention_dict = {
                'temporal_attentions': temporal_attentions,
                'static_feature': H_static,
                'dynamic_feature': H_dynamic_agg,
                'fused_feature': H_final,
                'adaptive_info': adaptive_info,
                'fusion_weights': fusion_weights,
                'cold_info': cold_info_dict
            }
            return pi, mu, theta, attention_dict

        if return_adaptive_info:
            if self.num_predict_types > 1:
                # 多类型输出
                outputs_list = [outputs[t] for t in self.predict_types]
                return tuple(outputs_list), H_static, H_dynamic_agg, adaptive_info
            return (pi, mu, theta), H_static, H_dynamic_agg, adaptive_info

        if self.num_predict_types > 1:
            # 多类型输出（向后兼容）
            outputs_list = [outputs[t] for t in self.predict_types]
            return tuple(outputs_list), H_static, H_dynamic_agg
        return pi, mu, theta, H_static, H_dynamic_agg

    def get_attention_analysis(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                               OD=None, kg_embed=None, grid_ids=None):
        """
        导出注意力分析结果 (用于可解释性)
        """
        self.eval()
        with torch.no_grad():
            pi, mu, theta, attention_dict = self.forward(
                X, A_spatial, A_distance, A_crime, A_hypergraph, OD,
                kg_embed=kg_embed,
                return_attention=True
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
            },
            'fusion_weights': attention_dict.get('fusion_weights', None),
            'cold_info': attention_dict.get('cold_info', {})
        }

        if grid_ids is not None:
            analysis['grid_ids'] = grid_ids

        return analysis

    def get_feature_importance(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                               OD=None, kg_embed=None, epsilon=0.01, feature_names=None):
        """
        特征重要性分析 (基于输入扰动方法)

        参数:
            X: 输入特征 (B, T, N, F) 或 (B, N, F)
            kg_embed: KG嵌入 (N, D_kg)
            epsilon: 扰动幅度
            feature_names: 特征名称列表（可选）
        """
        self.eval()

        # 确保输入是tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=A_spatial.device)
        else:
            X = X.to(A_spatial.device)

        with torch.no_grad():
            # 原始预测
            if kg_embed is not None:
                outputs_orig = self.forward(X, A_spatial, A_distance, A_crime,
                                            A_hypergraph, OD, kg_embed=kg_embed)
            else:
                outputs_orig = self.forward(X, A_spatial, A_distance, A_crime,
                                            A_hypergraph, OD)
            # 处理双任务输出
            if isinstance(outputs_orig[0], tuple):
                pi_orig, mu_orig = outputs_orig[0][0], outputs_orig[0][1]
            else:
                pi_orig, mu_orig = outputs_orig[0], outputs_orig[1]
            pred_orig = ((1 - pi_orig) * mu_orig).sum().item()

            print(f"Computing feature importance for {X.shape[-1]} features...")
            print(f"Original prediction sum: {pred_orig:.4f}")

            # 对每个特征维度进行扰动
            importance = np.zeros(X.shape[-1])
            for i in range(X.shape[-1]):
                if i % 10 == 0:
                    print(f"  Feature {i}/{X.shape[-1]}...")

                X_perturbed = X.clone()
                X_perturbed[..., i] += epsilon

                if kg_embed is not None:
                    outputs_perturbed = self.forward(X_perturbed, A_spatial, A_distance,
                                                     A_crime, A_hypergraph, OD, kg_embed=kg_embed)
                else:
                    outputs_perturbed = self.forward(X_perturbed, A_spatial, A_distance,
                                                     A_crime, A_hypergraph, OD)

                if isinstance(outputs_perturbed[0], tuple):
                    pi_new, mu_new = outputs_perturbed[0][0], outputs_perturbed[0][1]
                else:
                    pi_new, mu_new = outputs_perturbed[0], outputs_perturbed[1]
                pred_new = ((1 - pi_new) * mu_new).sum().item()

                importance[i] = abs(pred_new - pred_orig)

        print("\nFeature importance computation complete!")

        # 打印统计信息
        print("\n" + "="*60)
        print("特征重要性分析结果")
        print("="*60)

        # 排序
        sorted_idx = np.argsort(importance)[::-1]

        # 高贡献特征（Top 10）
        print("\n高贡献特征 (Top 10):")
        for i, idx in enumerate(sorted_idx[:10]):
            name = feature_names[idx] if feature_names else f"Feature_{idx}"
            print(f"  {i+1:2d}. {name:20s}: {importance[idx]:.6f}")

        # 低贡献特征（Bottom 10）
        print("\n低贡献特征 (Bottom 10):")
        for i, idx in enumerate(sorted_idx[-10:]):
            name = feature_names[idx] if feature_names else f"Feature_{idx}"
            print(f"  {i+1:2d}. {name:20s}: {importance[idx]:.6f}")

        # 统计接近0的特征数
        near_zero = np.sum(importance < 0.001)
        print(f"\n接近0的特征数 (< 0.001): {near_zero}/{len(importance)} ({near_zero/len(importance)*100:.1f}%)")

        # 静态特征 vs 动态特征（假设前24维是静态特征）
        if len(importance) >= 24:
            static_imp = importance[:24].mean()
            dynamic_imp = importance[24:].mean() if len(importance) > 24 else 0
            print(f"\n静态特征平均贡献: {static_imp:.6f}")
            print(f"动态特征平均贡献: {dynamic_imp:.6f}")

        print("="*60)

        return importance


# ================================
# 9. ZINB损失函数
# ================================
def zinb_loss(y_true, pi, mu, theta, sample_weights=None, pi_temperature=0.7):
    """零膨胀负二项分布损失（支持样本权重，改进版）

    sample_weights: (B, N) 或 (B, 1)，高频区域权重更高
    """
    eps = 1e-8

    # clamp避免数值爆炸
    pi = torch.clamp(pi, eps, 1 - eps)
    mu = torch.clamp(mu, eps, 1e6)
    theta = torch.clamp(theta, eps, 1e6)
    # 应用温度系数减少pi对低值的偏向
    pi = torch.pow(pi, pi_temperature)

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

    # 应用样本权重
    if sample_weights is not None:
        # 确保维度匹配
        if sample_weights.dim() == 2 and result.dim() == 2:
            # sample_weights: (B, 1) -> (B, N)
            if sample_weights.shape[1] == 1:
                sample_weights = sample_weights.expand(-1, result.shape[1])
        result = result * sample_weights
        return -torch.mean(result)

    return -torch.mean(result)


def hit_rate_loss(y_true, y_pred, k_percent=0.1):
    """
    Top-k Hit Rate 损失 - 直接优化实战指标

    原理：惩罚模型未能将真实高犯罪区域排进Top-k的情况
    y_true: (B, N) 真实犯罪数
    y_pred: (B, N) 预测犯罪数
    k_percent: Top-k百分比 (默认10%)
    """
    B, N = y_true.shape
    k = max(int(N * k_percent), 1)

    # 对每个样本，找出真实Top-k区域
    _, top_k_true_idx = torch.topk(y_true, k, dim=-1)  # (B, k)

    # 创建mask：真实Top-k区域=1，其他=0
    mask = torch.zeros_like(y_true)
    mask.scatter_(1, top_k_true_idx, 1.0)

    # 计算预测在真实Top-k区域的排名损失
    # 如果真实Top-k区域的预测值低，则损失大
    pred_in_topk = y_pred * mask  # 只保留真实Top-k区域的预测值

    # 惩罚：真实Top-k区域的预测值应该高于其他区域
    # 使用hinge loss: max(0, margin - (pred_topk - pred_other))
    margin = 1.0
    pred_other = y_pred * (1 - mask)  # 非Top-k区域的预测

    # 对每个样本，计loss算Top-k区域应该比第k+1大的惩罚
    loss = 0
    for i in range(B):
        topk_vals = pred_in_topk[i][top_k_true_idx[i]]  # (k,)
        # 第k+1大的预测值（作为阈值）
        kth_plus_one_val = torch.sort(y_pred[i], descending=True)[0][k] if k < N else topk_vals.min()
        # 惩罚真实Top-k中预测值低于阈值的
        loss += torch.clamp(margin - (topk_vals - kth_plus_one_val), min=0).mean()

    return loss / B


def focal_zinb_loss(y_true, pi, mu, theta, alpha=0.25, gamma=2.0):
    """
    增强型ZINB Loss - 保留备用（当前训练使用标准ZINB损失）

    alpha: 平衡因子（增加正样本权重）
    gamma: 聚焦因子（降低易分样本权重）
    """
    eps = 1e-8
    pi = torch.clamp(pi, eps, 1 - eps)
    mu = torch.clamp(mu, eps, 1e6)
    theta = torch.clamp(theta, eps, 1e6)

    # 标准ZINB损失计算
    t1 = torch.lgamma(theta + y_true)
    t2 = torch.lgamma(theta)
    t3 = torch.lgamma(y_true + 1)

    log_nb = (
        t1 - t2 - t3 +
        theta * (torch.log(theta) - torch.log(theta + mu)) +
        y_true * (torch.log(mu) - torch.log(theta + mu))
    )

    log_zero_nb = theta * (torch.log(theta) - torch.log(theta + mu))
    zero_case = torch.log(pi + (1 - pi) * torch.exp(log_zero_nb) + eps)
    non_zero_case = torch.log(1 - pi + eps) + log_nb

    zinb_loss = -torch.where(y_true < 1e-6, zero_case, non_zero_case)

    # 动态权重
    # 对于正样本(y>0)，如果预测prob低(难分)，权重高
    # 对于负样本(y=0)，如果预测prob高(假阳性)，权重高
    prob = torch.exp(-zinb_loss)  # 模型置信度（近似的）

    # 正样本权重增加alpha倍
    weight = torch.where(y_true > 0, alpha, 1 - alpha)
    # 样本聚焦：降低易分样本权重
    focal_weight = weight * torch.pow(1 - prob, gamma)

    return (focal_weight * zinb_loss).mean()


# ================================
# 10. 训练函数
# ================================
def train_model(model, train_loader, val_loader,
                A_spatial, A_distance, A_hypergraph,
                device, epochs=100, lr=1e-3, weight_decay=1e-5,
                clip_norm=5.0, patience=15,
                kg_embed=None,
                crime_history_train=None, crime_history_val=None,
                loss_type='zinb',  # 损失类型: 'zinb'（使用标准ZINB损失）
                hitrate_weight=0.0,  # Hit Rate损失权重（当前未使用）
                focal_alpha=0.25, focal_gamma=0.0, pi_temperature=0.7):  # 增强型ZINB参数（当前未使用）
    """
    训练时空Transformer模型（单任务 - 暴力犯罪预测）

    参数说明:
        kg_embed: (N, D) 知识图谱嵌入
        crime_history_train/val: 历史犯罪序列，用于近重复效应建模
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_loss = float('inf')
    patience_counter = 0

    # 记录专家权重的缓冲区
    expert_weights_history = []
    complexity_history = []

    for epoch in range(epochs):
        # ========== 训练 ==========
        model.train()
        train_losses = []
        epoch_expert_weights = []
        epoch_complexity = []

        # 记录分支统计
        high_freq_ratios = []

        for batch_idx, batch in enumerate(train_loader):
            # 解包 batch（多任务模式: X, A_crime, OD, Y_violent, Y_property）
            X_batch, A_crime_batch, OD_batch, Y_v_batch, Y_p_batch = batch

            X_batch = X_batch.to(device)
            Y_v_batch = Y_v_batch.to(device)
            Y_p_batch = Y_p_batch.to(device) if Y_p_batch is not None else None
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)

            # 处理NaN
            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            optimizer.zero_grad()

            # 提取犯罪历史 (用于自适应融合)
            crime_history = X_batch[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            B = X_batch.shape[0]

            # 扩展 batch 维度
            kg_embed_batch = kg_embed.unsqueeze(0).expand(B, -1, -1)  # (B, N, D_kg)

            # 构造crime_history_dict（多任务：violent + property）
            crime_history_dict = {
                'violent': Y_v_batch.unsqueeze(-1).expand(-1, -1, 7) if Y_v_batch.dim() == 2 else Y_v_batch,
                'property': Y_p_batch.unsqueeze(-1).expand(-1, -1, 7) if Y_p_batch is not None and Y_p_batch.dim() == 2 else Y_p_batch if Y_p_batch is not None else torch.zeros_like(Y_v_batch).unsqueeze(-1).expand(-1, -1, 7)
            }

            # 前向传播（多任务 - 同时预测暴力和财产犯罪）
            outputs = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                kg_embed=kg_embed_batch,
                crime_history=crime_history,
                crime_history_dict=crime_history_dict,
                return_adaptive_info=True
            )

            # 解包输出（多任务模式）
            # outputs: ((pi_v, mu_v, theta_v), (pi_p, mu_p, theta_p)), H_static, H_dynamic, adaptive_info
            predictions, h_static, h_dynamic, adaptive_info = outputs
            (pi_v, mu_v, theta_v), (pi_p, mu_p, theta_p) = predictions

            # 数值稳定
            mu = torch.clamp(mu, max=100)
            theta = torch.clamp(theta, max=100)

            # 记录分支统计信息
            if adaptive_info is not None:
                epoch_expert_weights.append(adaptive_info['expert_weights'].mean(dim=(0,1)).detach().cpu())
                epoch_complexity.append(adaptive_info['complexity_score'].mean().item())
                high_freq_ratios.append(adaptive_info.get('high_freq_ratio', 0.0))

            # 标准ZINB损失（主任务 - 暴力犯罪）
            loss_zinb_v = zinb_loss(Y_v_batch, pi_v, mu_v, theta_v, pi_temperature=pi_temperature)

            # Count Regression Loss: 直接优化预测期望值
            pred_v = (1 - pi_v) * mu_v
            loss_reg_v = F.smooth_l1_loss(pred_v, Y_v_batch)

            # 辅助任务损失（财产犯罪）
            if Y_p_batch is not None:
                loss_zinb_p = zinb_loss(Y_p_batch, pi_p, mu_p, theta_p, pi_temperature=pi_temperature)
                pred_p = (1 - pi_p) * mu_p
                loss_reg_p = F.smooth_l1_loss(pred_p, Y_p_batch)
                loss_aux = loss_zinb_p + 0.3 * loss_reg_p
            else:
                loss_aux = 0.0

            # 联合损失（主任务 + lambda_aux * 辅助任务）
            lambda_aux = 0.3
            loss = (loss_zinb_v + 0.3 * loss_reg_v) + lambda_aux * loss_aux
            loss_str = f"ZINB_v={loss_zinb_v.item():.3f} Reg_v={loss_reg_v.item():.3f} Aux={loss_aux.item() if isinstance(loss_aux, torch.Tensor) else 0.0:.3f}"

            # 记录各损失分量（每10个batch）
            if batch_idx % 10 == 0:
                print(f"    Loss: {loss_str}")

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            train_losses.append(loss.item())

        # ========== 验证 ==========
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                # 解包 batch（单任务模式: X, A_crime, OD, Y）
                X_batch, A_crime_batch, OD_batch, Y_batch = batch

                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                A_crime_batch = A_crime_batch.to(device)
                OD_batch = OD_batch.to(device)

                if torch.isnan(A_crime_batch).any():
                    A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

                # 提取犯罪历史
                crime_history = X_batch[:, :, :, -7:]
                crime_history = crime_history[:, :, :, 0]
                B = X_batch.shape[0]

                # 扩展 batch 维度
                kg_embed_batch = kg_embed.unsqueeze(0).expand(B, -1, -1)  # (B, N, D_kg)

                # 前向传播（单任务 - 只预测暴力犯罪）
                outputs = model(
                    X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                    kg_embed=kg_embed_batch,
                    crime_history=crime_history
                )

                # 解包输出（单任务模式: (pi, mu, theta, H_static, H_dynamic)）
                pi, mu, theta, _, _ = outputs

                mu = torch.clamp(mu, max=100)
                theta = torch.clamp(theta, max=100)

                # 验证损失（单任务 - 暴力犯罪）
                loss = zinb_loss(Y_batch, pi, mu, theta, pi_temperature=pi_temperature)
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        # 计算并打印分支统计
        if epoch_expert_weights:
            avg_weights = torch.stack(epoch_expert_weights).mean(dim=0)
            avg_complexity = np.mean(epoch_complexity)
            avg_high_freq = np.mean(high_freq_ratios) if high_freq_ratios else 0.0
            print(f"Epoch {epoch+1:3d} | Train Loss: {mean_train:.4f} | Val Loss: {mean_val:.4f} | "
                  f"Density: {avg_complexity:.3f} | "
                  f"Branch: [HighFreq:{avg_weights[0]:.2f} LowFreq:{avg_weights[1]:.2f}] "
                  f"HighRatio:{avg_high_freq:.1%}")
            # 保存到历史
            expert_weights_history.append(avg_weights.numpy())
            complexity_history.append(avg_complexity)
        else:
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


def test_model(model, test_loader, A_spatial, A_distance, A_hypergraph, device, kg_embed):
    """测试模型（多任务模式 - 但只评估暴力犯罪分支）"""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            # 解包 batch（多任务模式: X, A_crime, OD, Y_violent, Y_property）
            X_batch, A_crime_batch, OD_batch, Y_v_batch, Y_p_batch = batch
            X_batch = X_batch.to(device)
            Y_v_batch = Y_v_batch.to(device)
            A_crime_batch = A_crime_batch.to(device)
            OD_batch = OD_batch.to(device)

            if torch.isnan(A_crime_batch).any():
                A_crime_batch = torch.nan_to_num(A_crime_batch, nan=0.0)

            # 提取犯罪历史
            crime_history = X_batch[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]
            B = X_batch.shape[0]
            kg_embed_batch = kg_embed.unsqueeze(0).expand(B, -1, -1)

            # 构造crime_history_dict
            crime_history_dict = {
                'violent': Y_v_batch.unsqueeze(-1).expand(-1, -1, 7) if Y_v_batch.dim() == 2 else Y_v_batch,
                'property': Y_p_batch.unsqueeze(-1).expand(-1, -1, 7) if Y_p_batch is not None and Y_p_batch.dim() == 2 else torch.zeros_like(Y_v_batch).unsqueeze(-1).expand(-1, -1, 7)
            }

            # 调用模型（多任务模式）
            outputs = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph, OD_batch,
                kg_embed=kg_embed_batch,
                crime_history=crime_history,
                crime_history_dict=crime_history_dict
            )

            # 解包输出（多任务模式: ((pi_v, mu_v, theta_v), (pi_p, mu_p, theta_p)), H_static, H_dynamic）
            predictions, _, _ = outputs  # (predictions, H_static, H_dynamic)
            (pi_v, mu_v, theta_v), (pi_p, mu_p, theta_p) = predictions

            # 只使用暴力犯罪分支进行评估
            pred = torch.clamp((1 - pi_v) * mu_v, min=0)
            preds.append(pred.cpu().numpy())
            targets.append(Y_v_batch.cpu().numpy())

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

    # 打印多类型近重复效应参数 (如果有)
    if hasattr(model, 'multitype_nr') and model.multitype_nr is not None:
        decay_params = model.multitype_nr.get_decay_params()
        print(f"\n多类型近重复效应参数:")
        for ctype, params in decay_params.items():
            print(f"  {ctype}: α={params['spatial_decay']:.4f}, β={params['temporal_decay']:.4f}")

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
    # 加载特征（优先使用融入经济指标的版本）
    if os.path.exists("data/processed/X_with_econ.npy"):
        X = np.load("data/processed/X_with_econ.npy")
        print("使用融入经济指标的特征: X_with_econ.npy")
    else:
        X = np.load("data/processed/X.npy")
        print("使用原始特征: X.npy")
    print(f"X shape: {X.shape}")
    Y = np.load("data/processed/Y.npy")   # (T', N)
    OD = np.load("data/processed/dynamic_od_flow.npy")  # (T', N, 4)
    OD = np.log1p(OD)

    A_spatial = np.load("data/processed/adj_adaptive.npy")
    A_distance = np.load("data/processed/adj_distance.npy")
    # 使用暴力犯罪传播图（与预测目标一致）
    A_crime_dynamic = np.load("data/processed/adj_crime_violent.npy")
    # 加载财产犯罪传播图（用于双犯罪融合模块）
    A_crime_property = np.load("data/processed/adj_crime_property.npy")
    A_hypergraph = np.load("data/processed/adj_hypergraph.npy")

    
    # 加载知识图谱嵌入 (KG-enhanced版本)
    kg_embed_path = "data/processed/kg_embeddings.npy"
    if os.path.exists(kg_embed_path):
        kg_embed = np.load(kg_embed_path)
        print(f"Loaded KG embeddings: {kg_embed.shape}")
    else:
        # 如果KG嵌入不存在，使用零矩阵占位（将自动退化为非KG模式）
        N = Y.shape[1]  # 网格数量
        kg_embed = np.zeros((N, 32), dtype=np.float32)
        print(f"KG embeddings not found, using zero placeholder: {kg_embed.shape}")
    # ========== 构建窗口 ==========
    window = cfg.training.window
    crime_lag = cfg.training.crime_lag
    offset = window - crime_lag

    X_window = []
    Y_window = []

    for i in range(len(X) - offset):
        X_window.append(X[i:i+offset])
        Y_window.append(Y[i+offset])

    X_window = np.stack(X_window, axis=0).astype(np.float32)
    Y_window = np.stack(Y_window, axis=0).astype(np.float32)

    # Y_window: (num_samples, N, 2) -> 多任务模式（暴力+财产）
    if Y_window.ndim == 3 and Y_window.shape[-1] == 2:
        Y_window_violent = Y_window[..., 0]   # (num_samples, N) - 暴力犯罪
        Y_window_property = Y_window[..., 1]  # (num_samples, N) - 财产犯罪（辅助任务）
        print("多任务模式：暴力犯罪（主）+ 财产犯罪（辅）")
    else:
        Y_window_violent = Y_window
        Y_window_property = None

    print(f"X_window: {X_window.shape}")
    print(f"Y_window: {Y_window.shape}")

    # ========== 划分数据集 ==========
    num_samples = X_window.shape[0]
    train_ratio = 0.7
    val_ratio = 0.15

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    X_train, Y_train = X_window[:train_end], Y_window_violent[:train_end]
    X_val, Y_val = X_window[train_end:val_end], Y_window_violent[train_end:val_end]
    X_test, Y_test = X_window[val_end:], Y_window_violent[val_end:]

    # 财产犯罪标签（辅助任务）
    if Y_window_property is not None:
        Y_property_train = Y_window_property[:train_end]
        Y_property_val = Y_window_property[train_end:val_end]
        Y_property_test = Y_window_property[val_end:]
    else:
        Y_property_train = Y_property_val = Y_property_test = None

    # A_crime_violent: (700, N, N) 与 Y_window (700, N) 时间对齐，直接使用
    A_crime_train = A_crime_dynamic[:train_end]
    A_crime_val = A_crime_dynamic[train_end:val_end]
    A_crime_test = A_crime_dynamic[val_end:num_samples]

    # A_property 同样划分
    A_property_train = A_crime_property[:train_end]
    A_property_val = A_crime_property[train_end:val_end]
    A_property_test = A_crime_property[val_end:num_samples]

    # 财产犯罪图划分（用于双犯罪融合）
    A_property_train = A_crime_property[:train_end]
    A_property_val = A_crime_property[train_end:val_end]
    A_property_test = A_crime_property[val_end:num_samples]

    # OD: (730, N, 4) 需要与 Y_window 对齐
    # Y_window[i] 对应时间 i+offset，所以 OD 也从 offset 开始
    OD_aligned = OD[offset:offset+num_samples]  # (700, N, 4)
    OD_train = OD_aligned[:train_end]
    OD_val = OD_aligned[train_end:val_end]
    OD_test = OD_aligned[val_end:]

    # ========== DataLoader ==========
    batch_size = cfg.training.batch_size

    # 多任务数据集（暴力犯罪主任务 + 财产犯罪辅助任务）
    train_dataset = CrimeDataset(X_train, Y_train, A_crime_train, OD_train,
                                  A_property=A_property_train, Y_property=Y_property_train)
    val_dataset = CrimeDataset(X_val, Y_val, A_crime_val, OD_val,
                                A_property=A_property_val, Y_property=Y_property_val)
    test_dataset = CrimeDataset(X_test, Y_test, A_crime_test, OD_test, A_property=A_property_test, Y_property=Y_property_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 模型初始化 ==========
    static_dim = cfg.dataset.static_dim
    dynamic_dim = X.shape[2] - static_dim
    num_nodes = X.shape[1]

    kg_embed = torch.tensor(kg_embed, dtype=torch.float32).to(device)

    print(f"\n模型参数:")
    print(f"  - static_dim: {static_dim}")
    print(f"  - dynamic_dim: {dynamic_dim}")
    print(f"  - num_nodes: {num_nodes}")

    model = SpatioTemporalTransformer(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        kg_dim=32,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_temporal_layers=cfg.model.num_temporal_layers,
        num_spatial_layers=cfg.model.num_spatial_layers,
        dropout=cfg.model.dropout,
        num_nodes=num_nodes,
        use_multitype_nr=True,   # 启用多类型近重复效应（但只用于暴力犯罪）
        distance_matrix=None,
        predict_property=False,  # 禁用双任务 - 单任务（暴力犯罪）
        predict_public=False
    ).to(device)

    print(f"\n模型配置:")
    print(f"  - 任务类型: 单任务（暴力犯罪预测）")
    print(f"  - 预测类型: {model.predict_types}")
    print(f"  - 近重复效应模块: {'启用' if model.use_multitype_nr else '禁用'}")
    # 图矩阵转tensor
    A_spatial = torch.tensor(A_spatial, dtype=torch.float32).to(device)
    A_distance = torch.tensor(A_distance, dtype=torch.float32).to(device)
    A_hypergraph = torch.tensor(A_hypergraph, dtype=torch.float32).to(device)

    # ========== 训练 ==========
    print("\n开始训练...")
    print("使用原始ZINB损失 + 单任务学习（暴力犯罪）")
    model = train_model(
        model, train_loader, val_loader,
        A_spatial, A_distance, A_hypergraph,
        device,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        clip_norm=cfg.training.clip_norm,
        kg_embed=kg_embed,
        # 使用原始ZINB损失
        loss_type='zinb',
        hitrate_weight=0.0,
        focal_alpha=0.25,
        focal_gamma=0.0
    )

    # ========== 测试 ==========
    # 加载最佳模型
    model.load_state_dict(torch.load("checkpoints/best_model_trans.pt"))
    pred_test, Y_test_all = test_model(
        model, test_loader, A_spatial, A_distance, A_hypergraph, device, kg_embed=kg_embed
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
            OD_sample, kg_embed=kg_embed
        )

        print(f"预测期望犯罪数: {analysis['prediction']['expected_crime'].shape}")
        print(f"时间注意力层数: {len(analysis['attention_weights']['temporal'])}")

        # 特征重要性
        importance = model.get_feature_importance(
            X_sample, A_spatial, A_distance, A_crime_sample, A_hypergraph,
            OD_sample, kg_embed=kg_embed
        )
        print(f"特征重要性: {importance.shape}")


    print("\n✅ 训练完成!")
