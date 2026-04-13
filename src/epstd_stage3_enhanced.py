"""
EP-STD Stage 3 Enhanced: 条件扩散模型 + 逻辑引导 (Logic-Guided Diffusion)
核心创新：
  1. Classifier Guidance: 在去噪过程中引入逻辑约束梯度引导
  2. T-Norm软逻辑: 将RAG规则转化为可微损失
  3. 支持压力测试(Masking Experiment)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math
import pickle


# ==================== T-Norm 软逻辑实现 ====================

class SoftLogic(nn.Module):
    """
    T-Norm软逻辑: 将布尔逻辑算子映射为连续可微函数

    规则示例: "商业高 ∧ 监控低 → 高风险"
    """

    @staticmethod
    def t_norm_product(a, b):
        """乘积T-Norm: T(a,b) = a * b"""
        return a * b

    @staticmethod
    def t_norm_godel(a, b):
        """Godel T-Norm: T(a,b) = min(a,b)"""
        return torch.min(a, b)

    @staticmethod
    def t_norm_lukasiewicz(a, b):
        """Lukasiewicz T-Norm: T(a,b) = max(0, a+b-1)"""
        return torch.clamp(a + b - 1, min=0)

    @staticmethod
    def implication_reichen(a, b):
        """Reichenbach蕴涵: I(a,b) = 1 - a + a*b"""
        return 1 - a + a * b

    @staticmethod
    def implication_goguen(a, b):
        """Goguen蕴涵: I(a,b) = min(1, b/a) if a>0 else 1"""
        return torch.where(a > 0, torch.clamp(b / (a + 1e-8), max=1), torch.ones_like(a))

    @staticmethod
    def negation(a):
        """标准否定: N(a) = 1 - a"""
        return 1 - a

    @classmethod
    def evaluate_rule(cls, premises, conclusion, implication='goguen'):
        """
        评估逻辑规则: premises → conclusion

        Args:
            premises: (B, N) 前提的真值度 [0,1]，例如 "商业高 ∧ 监控低"
            conclusion: (B, N) 结论的真值度 [0,1]，例如 "高风险预测"
            implication: 蕴涵算子类型

        Returns:
            satisfaction: (B, N) 规则满足度，越接近1表示满足度越高
            loss: 标量，逻辑损失 = 1 - satisfaction
        """
        if implication == 'reichen':
            impl = cls.implication_reichen(premises, conclusion)
        else:  # goguen
            impl = cls.implication_goguen(premises, conclusion)

        # 损失：不满足度
        loss = (1 - impl).mean()
        return impl, loss


class FuzzyRiskPrior(nn.Module):
    """
    模糊逻辑风险先验: 处理规则冲突

    将风险因子和防护因子映射到同一语义空间进行加权抵消
    """

    def __init__(self, num_risk_factors=5, num_protective_factors=5, hidden_dim=32):
        super().__init__()

        # 风险因子编码器 (商业密度、人流、夜间等)
        self.risk_encoder = nn.Sequential(
            nn.Linear(num_risk_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 防护因子编码器 (监控、照明、CPTED等)
        self.protective_encoder = nn.Sequential(
            nn.Linear(num_protective_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 可学习的权重 (RAG初始化)
        self.risk_weight = nn.Parameter(torch.tensor(1.0))
        self.protective_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, risk_factors, protective_factors, rag_weights=None):
        """
        Args:
            risk_factors: (N, num_risk) 风险因子
            protective_factors: (N, num_protective) 防护因子
            rag_weights: Dict{'risk': float, 'protective': float} RAG提供的先验权重

        Returns:
            risk_prior: (N,) 风险先验分数 [0,1]
        """
        # 编码因子
        risk_score = self.risk_encoder(risk_factors).squeeze(-1)  # (N,)
        protective_score = self.protective_encoder(protective_factors).squeeze(-1)  # (N,)

        # 应用RAG权重（如果提供）
        if rag_weights is not None:
            w_risk = rag_weights.get('risk', 1.0)
            w_prot = rag_weights.get('protective', 0.8)
        else:
            w_risk = torch.sigmoid(self.risk_weight)
            w_prot = torch.sigmoid(self.protective_weight)

        # 加权抵消: 风险 - 防护
        net_risk = torch.sigmoid(w_risk * risk_score - w_prot * protective_score)

        return net_risk


# ==================== 逻辑约束计算器 ====================

class LogicConstraintCalculator(nn.Module):
    """
    从RAG规则和特征计算逻辑约束
    """

    def __init__(self, static_feature_dim=24):
        super().__init__()
        self.static_dim = static_feature_dim

        # 特征索引定义（根据你的数据调整）
        self.feature_indices = {
            'commercial_poi': 0,      # 商业POI
            'traffic_poi': 1,         # 交通POI
            'road_density': 3,        # 道路密度
            'residential': 4,         # 住宅用地
            'lighting': 8,            # 夜间照明
            'camera': 9,              # 摄像头
            'cpted_surveillance': 10, # CPTED监护
        }

    def extract_risk_factors(self, static_features):
        """提取风险因子"""
        # 商业密度、道路密度、（1-照明）、（1-监控）
        commercial = static_features[:, self.feature_indices['commercial_poi']]
        road = static_features[:, self.feature_indices['road_density']]
        dark = 1 - static_features[:, self.feature_indices['lighting']]
        unmonitored = 1 - static_features[:, self.feature_indices['camera']]

        # 时间因子（假设有的话）
        time_factor = torch.ones_like(commercial) * 0.5  # 默认值

        return torch.stack([commercial, road, dark, unmonitored, time_factor], dim=1)

    def extract_protective_factors(self, static_features):
        """提取防护因子"""
        lighting = static_features[:, self.feature_indices['lighting']]
        camera = static_features[:, self.feature_indices['camera']]
        cpted = static_features[:, self.feature_indices['cpted_surveillance']]
        residential = static_features[:, self.feature_indices['residential']]

        # 人口密度（如果有）
        density = torch.ones_like(lighting) * 0.5

        return torch.stack([lighting, camera, cpted, residential, density], dim=1)

    def compute_logic_loss(self, pred_risk, static_features, rule_type='default'):
        """
        计算逻辑约束损失

        Args:
            pred_risk: (B, N) 模型预测的风险值
            static_features: (B, N, F) 静态环境特征
            rule_type: 规则类型

        Returns:
            loss: 标量
            rule_satisfaction: 规则满足度统计
        """
        B, N = pred_risk.shape

        # 提取因子
        risk_factors = self.extract_risk_factors(static_features.view(B*N, -1)).view(B, N, -1)
        protective_factors = self.extract_protective_factors(static_features.view(B*N, -1)).view(B, N, -1)

        # 计算环境风险先验（模糊逻辑）
        env_risk = torch.sigmoid(
            0.3 * risk_factors[:,:,0] +  # 商业
            0.2 * risk_factors[:,:,1] -  # 道路
            0.3 * protective_factors[:,:,0] -  # 照明
            0.2 * protective_factors[:,:,1]    # 监控
        )

        # T-Norm软逻辑评估规则: "环境高风险 → 预测高风险"
        # 前提: 环境风险高
        premise = (env_risk > 0.6).float() * env_risk  # 软阈值

        # 结论: 预测风险高
        conclusion = pred_risk

        # 双向约束
        # 1. 环境高风险但预测低 → 惩罚 (漏报)
        _, loss_under = SoftLogic.evaluate_rule(premise, conclusion, 'goguen')

        # 2. 环境低风险但预测高 → 惩罚 (误报)
        low_risk_premise = (env_risk < 0.3).float() * (1 - env_risk)
        low_risk_conclusion = 1 - conclusion  # 预测应该低
        _, loss_over = SoftLogic.evaluate_rule(low_risk_premise, low_risk_conclusion, 'goguen')

        # 总逻辑损失
        total_loss = loss_under + loss_over

        # 统计信息
        stats = {
            'under_detection_rate': (premise > 0.5).float().mean().item(),
            'over_detection_rate': (low_risk_premise > 0.5).float().mean().item(),
            'env_risk_mean': env_risk.mean().item()
        }

        return total_loss, stats


# ==================== 带逻辑引导的扩散调度器 ====================

class LogicGuidedDiffusionScheduler:
    """
    逻辑引导扩散调度器
    在DDPM基础上增加Classifier Guidance风格的逻辑约束引导
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 预计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0, t, noise):
        """前向扩散：q(x_t | x_0)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def denoise_step_with_guidance(self, model, x_t, t, env_emb, prototype_ids,
                                    static_features, logic_calculator,
                                    guidance_scale=1.0, adj_matrix=None):
        """
        带逻辑引导的单步去噪

        Args:
            model: 扩散模型
            x_t: 当前噪声状态
            t: 时间步
            env_emb: 环境嵌入
            prototype_ids: 原型ID
            static_features: 静态特征（用于计算逻辑约束）
            logic_calculator: 逻辑约束计算器
            guidance_scale: 逻辑引导强度 ω
            adj_matrix: 邻接矩阵

        Returns:
            x_{t-1}: 去噪后的状态
            pred_x_0: 预测的x_0
            pi: 零膨胀概率
        """
        x_t.requires_grad_(True)

        # 1. 标准扩散预测
        noise_pred, pi = model(x_t, t, env_emb, prototype_ids, adj_matrix)

        # 2. 计算逻辑约束梯度 (Classifier Guidance核心)
        if guidance_scale > 0 and static_features is not None:
            # 计算当前预测x_0
            sqrt_alpha = self.sqrt_alphas_cumprod[t[0]].item()
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]].item()
            pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

            # 计算逻辑损失
            logic_loss, _ = logic_calculator.compute_logic_loss(pred_x_0, static_features)

            # 计算梯度引导
            logic_grad = torch.autograd.grad(logic_loss, x_t)[0]

            # 应用引导: 噪声预测 + 逻辑梯度
            noise_pred = noise_pred - guidance_scale * logic_grad

        # 3. 标准去噪步骤
        x_t = x_t.detach()  # 不再需要梯度

        B = x_t.shape[0]
        alpha = self.alphas[t].view(-1, 1)
        alpha_cumprod = self.alphas_cumprod[t].view(-1, 1)
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1)

        # 重新计算引导后的x_0预测
        sqrt_alpha = self.sqrt_alphas_cumprod[t[0]].item()
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]].item()
        pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        # 计算x_{t-1}的均值
        pred_mean = (
            torch.sqrt(alpha_cumprod_prev) * self.betas[t].view(-1, 1) * pred_x_0 +
            torch.sqrt(alpha) * (1.0 - alpha_cumprod_prev) * x_t
        ) / (1.0 - alpha_cumprod)

        # 添加噪声（除了最后一步）
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t].view(-1, 1)
            pred_mean = pred_mean + torch.sqrt(variance) * noise

        return pred_mean, pred_x_0.detach(), pi.detach()


# ==================== 压力测试支持 ====================

def apply_masking_experiment(Y_train, mask_ratio=0.2, target='high_crime'):
    """
    压力测试：人为屏蔽高犯罪网格的历史数据

    Args:
        Y_train: (samples, N) 原始犯罪数据
        mask_ratio: 屏蔽比例
        target: 'high_crime' - 屏蔽高犯罪网格; 'random' - 随机屏蔽

    Returns:
        Y_masked: 屏蔽后的数据
        mask_indices: 被屏蔽的网格索引
    """
    samples, N = Y_train.shape
    n_mask = int(N * mask_ratio)

    # 计算每个网格的平均犯罪率
    avg_crime = Y_train.mean(axis=0)

    if target == 'high_crime':
        # 屏蔽高犯罪网格（模拟冷启动）
        mask_indices = np.argsort(avg_crime)[-n_mask:]
    else:
        # 随机屏蔽
        mask_indices = np.random.choice(N, n_mask, replace=False)

    # 创建屏蔽后的数据
    Y_masked = Y_train.copy()
    Y_masked[:, mask_indices] = 0  # 强制清零

    print(f"Masking Experiment: {n_mask} grids ({mask_ratio*100}%) masked")
    print(f"  Target: {target}")
    print(f"  Masked avg crime: {avg_crime[mask_indices].mean():.4f}")
    print(f"  Unmasked avg crime: {avg_crime[~np.isin(np.arange(N), mask_indices)].mean():.4f}")

    return Y_masked, mask_indices


def evaluate_cold_start_performance(pred_risk, y_true, mask_indices):
    """
    评估冷启动性能

    Args:
        pred_risk: (N,) 预测风险
        y_true: (N,) 真实风险
        mask_indices: 被屏蔽的网格索引

    Returns:
        metrics: 字典包含各项评估指标
    """
    # 只评估被屏蔽的网格
    pred_masked = pred_risk[mask_indices]
    true_masked = y_true[mask_indices]

    # 基础指标
    mae = np.abs(pred_masked - true_masked).mean()
    correlation = np.corrcoef(pred_masked, true_masked)[0, 1]

    # 热点召回（在被屏蔽网格中）
    n_hotspot = max(1, int(0.1 * len(mask_indices)))
    true_hotspots = np.argsort(true_masked)[-n_hotspot:]
    pred_hotspots = np.argsort(pred_masked)[-n_hotspot:]
    recall = len(set(true_hotspots) & set(pred_hotspots)) / len(true_hotspots)

    metrics = {
        'mae': mae,
        'correlation': correlation,
        'hotspot_recall': recall,
        'n_masked': len(mask_indices)
    }

    return metrics


# ==================== 双任务数据加载 ====================

def load_dual_task_data(data_dir='data/processed', crime_types=['violent', 'property']):
    """
    加载双任务犯罪数据（暴力+财产）

    Args:
        data_dir: 数据目录
        crime_types: 犯罪类型列表

    Returns:
        X: 共享的环境特征 (samples, N, F)
        Y_dict: {crime_type: Y_array} 每种犯罪的标签
    """
    print("Loading dual-task crime data...")

    # 加载共享特征
    X = np.load(f'{data_dir}/X.npy')

    # 加载各犯罪类型标签
    Y_dict = {}
    for crime_type in crime_types:
        y_path = f'{data_dir}/Y_{crime_type}.npy'
        if os.path.exists(y_path):
            Y_dict[crime_type] = np.load(y_path)
            print(f"  Loaded {crime_type}: {Y_dict[crime_type].shape}")
        else:
            print(f"  Warning: {y_path} not found, using default Y.npy")
            Y_dict[crime_type] = np.load(f'{data_dir}/Y.npy')

    return X, Y_dict


# ==================== 训练流程集成（双任务） ====================

def train_logic_guided_diffusion_dual(
    env_encoder,
    prototype_library,
    X_train,
    Y_violent,  # 暴力犯罪标签
    Y_property,  # 财产犯罪标签
    adj_matrix=None,
    epochs=100,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    use_logic_guidance=True,
    logic_warmup_epochs=20,
    guidance_scale_schedule='linear',
    task_weights=None  # {'violent': 1.0, 'property': 0.5}
):
    """
    训练带逻辑引导的双任务扩散模型

    Args:
        env_encoder: 环境编码器
        prototype_library: 原型库
        X_train: 共享特征 (samples, N, F)
        Y_violent: 暴力犯罪标签 (samples, N)
        Y_property: 财产犯罪标签 (samples, N)
        ...其他参数
    """
    from epstd_stage1 import EnvironmentEncoder
    from epstd_stage2 import PrototypeLibrary
    from epstd_stage3 import ConditionalRiskDiffusion

    print("="*70)
    print("Training Logic-Guided Diffusion Model (Dual-Task)")
    print(f"  Logic Guidance: {use_logic_guidance}")
    print(f"  Warmup Epochs: {logic_warmup_epochs}")
    print(f"  Data: {len(X_train)} samples, 2 years")
    print("="*70)

    # 冻结环境编码器
    env_encoder.eval()
    for param in env_encoder.parameters():
        param.requires_grad = False

    # 任务权重（默认财产犯罪权重较低，因为数据密集）
    if task_weights is None:
        task_weights = {'violent': 1.0, 'property': 0.5}

    # 创建双任务模型
    num_nodes = Y_violent.shape[1]

    # 修改：创建双任务模型
    class DualTaskDiffusion(nn.Module):
        """双任务扩散模型：暴力+财产"""

        def __init__(self, base_diffusion):
            super().__init__()
            self.base = base_diffusion

            # 为两个任务分别创建输出头
            self.violence_head = nn.Sequential(
                nn.Linear(base_diffusion.hidden_dim, base_diffusion.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(base_diffusion.hidden_dim // 2, 1)
            )

            self.property_head = nn.Sequential(
                nn.Linear(base_diffusion.hidden_dim, base_diffusion.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(base_diffusion.hidden_dim // 2, 1)
            )

            # 可学习的任务不确定性权重
            self.log_sigma_violent = nn.Parameter(torch.zeros(1))
            self.log_sigma_property = nn.Parameter(torch.zeros(1))

        def forward(self, x_t, t, env_emb, prototype_ids, adj_matrix=None):
            """
            前向传播，输出两个任务的噪声预测和零膨胀概率

            Returns:
                noise_violent, noise_property: 噪声预测
                pi_violent, pi_property: 零膨胀概率
            """
            # 使用基础扩散模型提取特征
            B, N = x_t.shape

            # 时间编码
            t_emb = self.base.time_mlp(t)

            # 环境编码
            h_env = self.base.env_encoder(env_emb)

            # 原型编码
            h_proto = self.base.prototype_embedding(prototype_ids)

            # 输入投影
            h = self.base.input_proj(x_t.unsqueeze(-1))

            # 图注意力层
            for gconv, fusion in zip(self.base.graph_layers, self.base.fusion_layers):
                if adj_matrix is not None:
                    h = gconv(h, adj_matrix)
                else:
                    h = gconv(h, torch.eye(N, device=h.device))

                t_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)
                fusion_input = torch.cat([h, h_env, t_expanded], dim=-1)
                h = fusion(fusion_input)

            # 任务特定输出
            noise_violent = self.violence_head(h).squeeze(-1)
            noise_property = self.property_head(h).squeeze(-1)

            # 零膨胀概率（共享特征，但分别预测）
            zero_input = torch.cat([h.mean(dim=1), env_emb.mean(dim=1)], dim=-1)
            pi_violent = self.base.zero_inflation_head(zero_input).expand(-1, N)
            pi_property = self.base.zero_inflation_head(zero_input).expand(-1, N)

            return (noise_violent, noise_property), (pi_violent, pi_property)

    # 基础模型
    base_model = ConditionalRiskDiffusion(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=4,
        time_dim=64,
        env_dim=64,
        num_prototypes=prototype_library.n_prototypes
    ).to(device)

    model = DualTaskDiffusion(base_model).to(device)

    scheduler = LogicGuidedDiffusionScheduler(num_timesteps=1000, device=device)
    logic_calculator = LogicConstraintCalculator(static_feature_dim=24).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 准备原型标签
    prototype_labels = np.load('data/processed/prototype_labels.npy')
    prototype_labels_tensor = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_violent_loss = 0
        total_property_loss = 0
        total_logic_loss = 0

        # 计算当前引导强度（退火策略）
        if epoch < logic_warmup_epochs:
            guidance_weight = 0.0
        else:
            progress = (epoch - logic_warmup_epochs) / (epochs - logic_warmup_epochs)
            if guidance_scale_schedule == 'linear':
                guidance_weight = progress * 1.0
            elif guidance_scale_schedule == 'cosine':
                guidance_weight = 0.5 * (1 - math.cos(progress * math.pi))
            else:
                guidance_weight = 1.0

        n_samples = len(Y_violent)
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            B = len(batch_idx)

            # 获取双任务数据
            y_violent_batch = torch.tensor(Y_violent[batch_idx], dtype=torch.float32).to(device)
            y_property_batch = torch.tensor(Y_property[batch_idx], dtype=torch.float32).to(device)

            x_static = X_train[batch_idx, :, :24]
            x_static_tensor = torch.tensor(x_static, dtype=torch.float32).to(device)

            # 编码环境（共享）
            with torch.no_grad():
                B_actual, N, F = x_static_tensor.shape
                x_static_flat = x_static_tensor.view(B_actual * N, F)
                env_emb_flat = env_encoder(x_static_flat)
                env_emb = env_emb_flat.view(B_actual, N, -1)

            # 采样时间步（双任务共享时间步）
            t = scheduler.sample_timesteps(B)

            # 加噪（分别加噪）
            noise_violent = torch.randn_like(y_violent_batch)
            noise_property = torch.randn_like(y_property_batch)
            x_t_violent = scheduler.add_noise(y_violent_batch, t, noise_violent)
            x_t_property = scheduler.add_noise(y_property_batch, t, noise_property)

            # 扩展原型标签
            proto_ids = prototype_labels_tensor.unsqueeze(0).expand(B, -1)

            # 前向传播 - 暴力任务
            (noise_pred_v, noise_pred_p), (pi_v, pi_p) = model(
                x_t_violent, t, env_emb, proto_ids, adj_matrix
            )

            # 扩散损失 - 暴力犯罪
            diffusion_loss_v = F.mse_loss(noise_pred_v, noise_violent)
            zero_mask_v = (y_violent_batch == 0).float()
            zinb_loss_v = F.binary_cross_entropy(pi_v, zero_mask_v)

            # 扩散损失 - 财产犯罪
            diffusion_loss_p = F.mse_loss(noise_pred_p, noise_property)
            zero_mask_p = (y_property_batch == 0).float()
            zinb_loss_p = F.binary_cross_entropy(pi_p, zero_mask_p)

            # 不确定性加权（双任务）
            precision_v = torch.exp(-model.log_sigma_violent)
            precision_p = torch.exp(-model.log_sigma_property)

            task_loss_v = precision_v * (diffusion_loss_v + 0.1 * zinb_loss_v) + model.log_sigma_violent
            task_loss_p = precision_p * (diffusion_loss_p + 0.1 * zinb_loss_p) + model.log_sigma_property

            # 逻辑约束损失（主要应用于暴力任务，因其稀疏）
            logic_loss = torch.tensor(0.0).to(device)
            if use_logic_guidance and guidance_weight > 0:
                sqrt_alpha = scheduler.sqrt_alphas_cumprod[t[0]].item()
                sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t[0]].item()

                # 暴力任务的逻辑约束（冷启动关键）
                pred_v_0 = (x_t_violent - sqrt_one_minus_alpha * noise_pred_v) / sqrt_alpha
                logic_loss_v, _ = logic_calculator.compute_logic_loss(pred_v_0, x_static_tensor)

                # 财产任务可选逻辑约束（数据密集，逻辑约束收益低）
                # pred_p_0 = (x_t_property - sqrt_one_minus_alpha * noise_pred_p) / sqrt_alpha
                # logic_loss_p, _ = logic_calculator.compute_logic_loss(pred_p_0, x_static_tensor)

                logic_loss = logic_loss_v * guidance_weight

            # 总损失（加权组合）
            loss = (task_weights['violent'] * task_loss_v +
                    task_weights['property'] * task_loss_p +
                    0.5 * logic_loss)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_violent_loss += task_loss_v.item()
            total_property_loss += task_loss_p.item()
            total_logic_loss += logic_loss.item()

        avg_loss = total_loss / (n_samples // batch_size + 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total: {avg_loss:.4f} | "
                  f"Violent: {total_violent_loss/(n_samples//batch_size+1):.4f} | "
                  f"Property: {total_property_loss/(n_samples//batch_size+1):.4f} | "
                  f"Logic: {total_logic_loss/(n_samples//batch_size+1):.4f} | "
                  f"GW: {guidance_weight:.3f} | "
                  f"σv: {torch.exp(model.log_sigma_violent).item():.3f} | "
                  f"σp: {torch.exp(model.log_sigma_property).item():.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'logic_calculator': logic_calculator.state_dict(),
                'task_weights': task_weights
            }, 'checkpoints/logic_guided_diffusion_dual_best.pt')

    print(f"\nTraining completed. Best loss: {best_loss:.4f}")

    return model, scheduler, logic_calculator


# ==================== 推理流程（双任务） ====================

class DualTaskLogicGuidedPredictor:
    """双任务带逻辑引导的预测器（暴力+财产）"""

    def __init__(self, model, scheduler, logic_calculator, env_encoder,
                 prototype_library, device='cuda'):
        self.model = model
        self.scheduler = scheduler
        self.logic_calculator = logic_calculator
        self.env_encoder = env_encoder
        self.prototype_library = prototype_library
        self.device = device

        self.model.eval()
        self.env_encoder.eval()

    @torch.no_grad()
    def predict(self, static_features, adj_matrix=None, num_samples=10,
                use_logic_guidance=True, guidance_scale=1.0,
                task='violent'):
        """
        生成风险预测（支持逻辑引导）

        Args:
            static_features: 静态特征
            adj_matrix: 邻接矩阵
            num_samples: 采样次数
            use_logic_guidance: 是否使用逻辑引导
            guidance_scale: 引导强度
            task: 'violent' 或 'property'

        Returns:
            risk_mean, risk_std, pi_mean
        """
        N = static_features.shape[0]

        # 编码环境
        x = torch.tensor(static_features, dtype=torch.float32).to(self.device)
        env_emb = self.env_encoder(x)

        # 查找原型
        prototype_ids = []
        for emb in env_emb.cpu().numpy():
            result = self.prototype_library.query(emb)
            prototype_ids.append(result['proto_id'])
        prototype_ids = torch.tensor(prototype_ids, dtype=torch.long).to(self.device)

        # 扩展为batch
        env_emb = env_emb.unsqueeze(0)
        prototype_ids = prototype_ids.unsqueeze(0)
        static_features_batch = torch.tensor(static_features, dtype=torch.float32).to(self.device).unsqueeze(0)

        # 双任务分别预测
        all_samples = []
        all_pis = []

        for _ in range(num_samples):
            x_t = torch.randn(1, N).to(self.device)

            for t in reversed(range(self.scheduler.num_timesteps)):
                t_tensor = torch.full((1,), t, dtype=torch.long).to(self.device)

                # 前向传播获取双任务预测
                (noise_v, noise_p), (pi_v, pi_p) = self.model(
                    x_t, t_tensor, env_emb, prototype_ids, adj_matrix
                )

                # 选择任务
                if task == 'violent':
                    noise_pred = noise_v
                    pi = pi_v
                else:
                    noise_pred = noise_p
                    pi = pi_p

                # 逻辑引导（仅对暴力任务）
                if use_logic_guidance and task == 'violent':
                    # 简化的引导实现
                    sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].item()
                    sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].item()
                    pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

                    # 计算逻辑梯度（简化版，实际应使用autograd）
                    logic_loss, _ = self.logic_calculator.compute_logic_loss(
                        pred_x_0, static_features_batch
                    )
                    # 这里简化处理，实际应在训练时存储引导方向

                # 标准去噪步骤
                sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].item()
                sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].item()
                pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

                if t > 0:
                    alpha = self.scheduler.alphas[t].item()
                    alpha_cumprod = self.scheduler.alphas_cumprod[t].item()
                    alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t].item()

                    pred_mean = (
                        np.sqrt(alpha_cumprod_prev) * self.scheduler.betas[t].item() * pred_x_0 +
                        np.sqrt(alpha) * (1.0 - alpha_cumprod_prev) * x_t
                    ) / (1.0 - alpha_cumprod)

                    variance = self.scheduler.posterior_variance[t].item()
                    noise = torch.randn_like(x_t)
                    x_t = pred_mean + np.sqrt(variance) * noise
                else:
                    x_t = pred_x_0

            all_samples.append(pred_x_0.cpu().numpy().squeeze())
            all_pis.append(pi.cpu().numpy().squeeze())

        risk_mean = np.mean(all_samples, axis=0)
        risk_std = np.std(all_samples, axis=0)
        pi_mean = np.mean(all_pis, axis=0)

        # 应用零膨胀调整
        risk_mean = risk_mean * (1 - pi_mean)

        return risk_mean, risk_std, pi_mean

    @torch.no_grad()
    def predict_dual(self, static_features, adj_matrix=None, num_samples=10,
                     use_logic_guidance=True, guidance_scale=1.0):
        """
        同时预测两个任务

        Returns:
            results: {
                'violent': (risk_mean, risk_std, pi_mean),
                'property': (risk_mean, risk_std, pi_mean)
            }
        """
        results = {}

        for task in ['violent', 'property']:
            risk_mean, risk_std, pi_mean = self.predict(
                static_features, adj_matrix, num_samples,
                use_logic_guidance if task == 'violent' else False,  # 仅暴力使用逻辑引导
                guidance_scale, task
            )
            results[task] = (risk_mean, risk_std, pi_mean)

        return results


def main_dual():
    """双任务主流程"""
    print("="*70)
    print("L-EPSTD Stage 3 Enhanced: Dual-Task Logic-Guided Diffusion")
    print("="*70)

    # 加载双任务数据
    data_dir = 'data/processed'
    X, Y_dict = load_dual_task_data(data_dir, crime_types=['violent', 'property'])

    Y_violent = Y_dict['violent']
    Y_property = Y_dict['property']

    print(f"Loaded dual-task data: X={X.shape}")
    print(f"  Violent crime: {Y_violent.shape}, sparsity={(Y_violent==0).mean():.2%}")
    print(f"  Property crime: {Y_property.shape}, sparsity={(Y_property==0).mean():.2%}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载环境编码器和原型库
    from epstd_stage1 import EnvironmentEncoder
    from epstd_stage2 import PrototypeLibrary

    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    env_encoder.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    env_encoder.eval()

    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']
    prototype_library.prototype_risk_dists = proto_data['prototype_risk_dists']

    # 划分数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_v_train, Y_v_test = Y_violent[:split_idx], Y_violent[split_idx:]
    Y_p_train, Y_p_test = Y_property[:split_idx], Y_property[split_idx:]

    # 训练双任务模型
    model, scheduler, logic_calculator = train_logic_guided_diffusion_dual(
        env_encoder, prototype_library,
        X_train, Y_v_train, Y_p_train,
        epochs=100, batch_size=16, device=device,
        use_logic_guidance=True,
        logic_warmup_epochs=20,
        task_weights={'violent': 1.0, 'property': 0.5}
    )

    print("\n" + "="*70)
    print("Dual-Task Training Completed!")
    print("="*70)


if __name__ == "__main__":
    print("L-EPSTD Stage 3 Enhanced: Dual-Task Logic-Guided Diffusion")
    print("="*70)
    print("Features:")
    print("  1. T-Norm Soft Logic")
    print("  2. Classifier Guidance in Denoising")
    print("  3. Dual-Task Learning (Violent + Property Crime)")
    print("  4. Uncertainty-Weighted Multi-Task Loss")
    print("="*70)

    # 运行双任务训练
    main_dual()
