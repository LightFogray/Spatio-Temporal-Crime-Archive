"""
EP-STD Stage 3: 条件扩散模型 (Conditional Diffusion Model)
目标：以环境嵌入为条件，生成风险分布预测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


def build_spatial_graph(grid_coords, k=8, sigma=1.0):
    """
    构建空间邻接矩阵（多图结构）

    Args:
        grid_coords: (N, 2) 网格坐标 [lat, lon]
        k: K近邻数量
        sigma: 距离衰减系数

    Returns:
        adj_matrix: (N, N) 归一化的邻接矩阵
    """
    N = len(grid_coords)

    # 计算欧氏距离矩阵
    coords = np.array(grid_coords)
    dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))

    # K近邻图：每个节点只连接最近的k个邻居
    adj_knn = np.zeros((N, N))
    for i in range(N):
        nearest_indices = np.argsort(dist_matrix[i])[1:k+1]  # 排除自身
        adj_knn[i, nearest_indices] = 1
        adj_knn[nearest_indices, i] = 1  # 对称

    # 距离加权：越近权重越高
    weights = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    adj_weighted = adj_knn * weights

    # 归一化（对称归一化）
    degree = adj_weighted.sum(axis=1, keepdims=True)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0

    adj_normalized = degree_inv_sqrt * adj_weighted * degree_inv_sqrt.T

    return torch.tensor(adj_normalized, dtype=torch.float32)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步t"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalRiskDiffusion(nn.Module):
    """
    条件风险扩散模型
    输入：噪声化的风险分布 + 环境条件
    输出：去噪后的风险分布
    """

    def __init__(
        self,
        num_nodes=1246,
        hidden_dim=128,
        num_layers=4,
        time_dim=64,
        env_dim=64,
        num_prototypes=10,
        dropout=0.1,
        num_graphs=3
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_graphs = num_graphs

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim)
        )

        # 环境条件编码
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 原型嵌入（可学习）
        self.prototype_embedding = nn.Embedding(num_prototypes, hidden_dim)

        # 输入投影：风险值 -> 隐藏空间
        self.input_proj = nn.Linear(1, hidden_dim)

        # 图注意力层（多图融合）
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout, num_graphs)
            for _ in range(num_layers)
        ])

        # 融合层：结合时间、环境、原型信息
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # 输出投影：隐藏空间 -> 风险值
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 零膨胀检测头：预测该网格是否为零犯罪
        self.zero_inflation_head = nn.Sequential(
            nn.Linear(hidden_dim + env_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_t, t, env_emb, prototype_ids, adj_matrix=None):
        """
        前向传播：预测噪声 epsilon

        Args:
            x_t: (B, N) 噪声化的风险值
            t: (B,) 时间步
            env_emb: (B, N, env_dim) 环境嵌入
            prototype_ids: (B, N) 每个网格的原型ID
            adj_matrix: (N, N) 空间邻接矩阵（可选）

        Returns:
            noise_pred: (B, N) 预测的噪声
            pi: (B, N) 零犯罪概率
        """
        B, N = x_t.shape

        # 时间编码 (B, hidden_dim)
        t_emb = self.time_mlp(t)

        # 环境编码 (B, N, hidden_dim)
        h_env = self.env_encoder(env_emb)

        # 原型编码 (B, N, hidden_dim)
        h_proto = self.prototype_embedding(prototype_ids)

        # 输入投影 (B, N, hidden_dim)
        h = self.input_proj(x_t.unsqueeze(-1))

        # 图注意力层
        # 确保adj_matrix是列表格式
        if adj_matrix is not None:
            if not isinstance(adj_matrix, list):
                adj_list = [adj_matrix]
            else:
                adj_list = adj_matrix
        else:
            # 无图结构时，使用自注意力
            adj_list = [torch.eye(N, device=h.device)]

        for i, (gconv, fusion) in enumerate(zip(self.graph_layers, self.fusion_layers)):
            # 图卷积（多图融合）
            h = gconv(h, adj_list)

            # 融合时间、环境、原型信息
            t_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
            fusion_input = torch.cat([h, h_env, t_expanded], dim=-1)  # (B, N, hidden_dim*3)
            h = fusion(fusion_input)

        # 预测噪声
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, N)

        # 预测零膨胀概率（结合最终特征和环境）
        zero_input = torch.cat([h.mean(dim=1), env_emb.mean(dim=1)], dim=-1)  # (B, hidden_dim + env_dim)
        pi = self.zero_inflation_head(zero_input)  # (B, 1)
        pi = pi.expand(-1, N)  # (B, N)

        return noise_pred, pi


class GraphAttentionLayer(nn.Module):
    """多图融合图注意力层"""

    def __init__(self, in_dim, out_dim, dropout=0.1, num_graphs=3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        # 使用双线性注意力代替拼接注意力，大幅减少内存
        self.attn_src = nn.Linear(out_dim, 1)
        self.attn_dst = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # 多图融合权重（可学习）
        self.num_graphs = num_graphs
        if num_graphs > 1:
            self.graph_fusion = nn.Sequential(
                nn.Linear(num_graphs, num_graphs),
                nn.Softmax(dim=-1)
            )

    def forward(self, h, adj_list):
        """
        h: (B, N, in_dim)
        adj_list: 邻接矩阵列表 [(N, N), ...] 或单个 (N, N)
        """
        B, N, _ = h.shape

        # 线性变换
        Wh = self.W(h)  # (B, N, out_dim)

        # 内存高效的注意力计算
        attn_src = self.attn_src(Wh).squeeze(-1)  # (B, N)
        attn_dst = self.attn_dst(Wh).squeeze(-1)  # (B, N)
        e = torch.tanh(attn_src.unsqueeze(2) + attn_dst.unsqueeze(1))  # (B, N, N)

        # 确保adj_list是列表
        if not isinstance(adj_list, list):
            adj_list = [adj_list]

        # 多图融合
        if len(adj_list) > 1 and self.num_graphs > 1:
            # 计算每个图的注意力
            attn_list = []
            for adj in adj_list:
                adj_expanded = adj.unsqueeze(0).expand(B, -1, -1)
                e_masked = e.masked_fill(adj_expanded == 0, float('-inf'))
                attn = F.softmax(e_masked, dim=-1)
                attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
                attn_list.append(attn)

            # 可学习的图权重 (B, N, num_graphs)
            graph_weights = self.graph_fusion(
                torch.stack([a.sum(dim=-1) for a in attn_list], dim=-1)
            )  # (B, N, num_graphs)

            # 加权融合
            attn_weights = sum(
                w.unsqueeze(-1) * a
                for w, a in zip(graph_weights.unbind(dim=-1), attn_list)
            )
        else:
            # 单图
            adj_expanded = adj_list[0].unsqueeze(0).expand(B, -1, -1)
            attn_weights = F.softmax(e.masked_fill(adj_expanded == 0, float('-inf')), dim=-1)
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # 聚合邻居信息
        h_new = torch.bmm(attn_weights, Wh)  # (B, N, out_dim)
        h_new = self.dropout(h_new)

        # 残差连接
        return F.gelu(h_new + Wh)


class DiffusionScheduler:
    """扩散调度器：管理加噪和去噪过程"""

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
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
        # 确保预计算张量在正确的设备上
        device = x_0.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        sqrt_alpha = sqrt_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alphas[t].view(-1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size, device):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def denoise_step(self, model, x_t, t, env_emb, prototype_ids, adj_matrix=None):
        """单步去噪：p(x_{t-1} | x_t)"""
        B = x_t.shape[0]
        device = x_t.device

        # 确保预计算张量在正确的设备上
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        betas = self.betas.to(device)
        posterior_variance = self.posterior_variance.to(device)

        # 预测噪声
        noise_pred, pi = model(x_t, t, env_emb, prototype_ids, adj_matrix)

        # 计算x_0预测
        sqrt_alpha = sqrt_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alphas[t].view(-1, 1)
        pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        # 计算x_{t-1}的均值
        alpha = alphas[t].view(-1, 1)
        alpha_cumprod_t = alphas_cumprod[t].view(-1, 1)
        alpha_cumprod_prev_t = alphas_cumprod_prev[t].view(-1, 1)

        pred_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * betas[t].view(-1, 1) * pred_x_0 +
            torch.sqrt(alpha) * (1.0 - alpha_cumprod_prev_t) * x_t
        ) / (1.0 - alpha_cumprod_t)

        # 添加噪声（除了最后一步）
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = posterior_variance[t].view(-1, 1)
            pred_mean = pred_mean + torch.sqrt(variance) * noise

        return pred_mean, pred_x_0, pi


class EPSTDPredictor:
    """
    EP-STD预测器：整合环境编码、原型库、扩散模型
    """

    def __init__(
        self,
        env_encoder,
        prototype_library,
        diffusion_model,
        scheduler,
        device='cuda',
        y_mean=0.0,
        y_std=1.0
    ):
        self.env_encoder = env_encoder
        self.prototype_library = prototype_library
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std

        self.env_encoder.eval()
        self.diffusion_model.eval()

        # 加载多图结构
        self.graph_list = []
        for graph_name in ['adj_adaptive', 'adj_distance', 'adj_crime_dynamic_gaussian']:
            graph_path = f'data/processed/{graph_name}.npy'
            if os.path.exists(graph_path):
                graph = torch.tensor(np.load(graph_path), dtype=torch.float32).to(device)
                self.graph_list.append(graph)
                print(f"Predictor loaded {graph_name}")

    @torch.no_grad()
    def predict(self, static_features, adj_matrix=None, num_samples=10):
        """
        生成风险预测

        Args:
            static_features: (N, F) 静态环境特征
            adj_matrix: (N, N) 空间邻接矩阵
            num_samples: 采样次数（用于估计不确定性）

        Returns:
            risk_mean: (N,) 预测风险均值
            risk_std: (N,) 预测风险标准差
            pi: (N,) 零犯罪概率
        """
        N = static_features.shape[0]

        # 编码环境
        x = torch.tensor(static_features, dtype=torch.float32).to(self.device)
        env_emb = self.env_encoder(x)  # (N, env_dim)

        # 查找原型
        prototype_ids = []
        for emb in env_emb.cpu().numpy():
            result = self.prototype_library.query(emb)
            prototype_ids.append(result['proto_id'])
        prototype_ids = torch.tensor(prototype_ids, dtype=torch.long).to(self.device)

        # 扩展为batch
        env_emb = env_emb.unsqueeze(0)  # (1, N, env_dim)
        prototype_ids = prototype_ids.unsqueeze(0)  # (1, N)

        # 多次采样去噪
        all_samples = []
        all_pis = []

        for _ in range(num_samples):
            # 从噪声开始
            x_t = torch.randn(1, N).to(self.device)

            # DDPM去噪：必须连续步骤，不能跳步
            for t in reversed(range(self.scheduler.num_timesteps)):
                t_tensor = torch.full((1,), t, dtype=torch.long).to(self.device)
                # 使用多图列表
                adj_for_model = self.graph_list if self.graph_list else adj_matrix
                x_t, pred_x_0, pi = self.scheduler.denoise_step(
                    self.diffusion_model, x_t, t_tensor,
                    env_emb, prototype_ids, adj_for_model
                )
                # 裁剪确保非负
                pred_x_0 = torch.clamp(pred_x_0, min=0)

            all_samples.append(pred_x_0.cpu().numpy().squeeze())
            all_pis.append(pi.cpu().numpy().squeeze())

        # 聚合多次采样结果
        risk_mean = np.mean(all_samples, axis=0)
        risk_std = np.std(all_samples, axis=0)
        pi_mean = np.mean(all_pis, axis=0)

        # 反归一化
        risk_mean = risk_mean * self.y_std + self.y_mean
        risk_std = risk_std * self.y_std

        # 应用零膨胀调整 - 使用阈值而非乘法，避免过度抑制
        zero_threshold = 0.7  # 降低阈值，让更多区域有预测值
        risk_mean = np.where(pi_mean > zero_threshold, 0, risk_mean)

        # 确保风险值非负
        risk_mean = np.maximum(risk_mean, 0)

        return risk_mean, risk_std, pi_mean


def train_diffusion_model(
    env_encoder,
    prototype_library,
    X_train,
    Y_train,
    adj_matrix=None,
    epochs=100,
    batch_size=32,
    lr=1e-4,
    device='cuda'
):
    """
    训练扩散模型

    Args:
        env_encoder: 预训练的环境编码器
        prototype_library: 原型库
        X_train: (samples, N, F) 训练数据
        Y_train: (samples, N) 风险标签
        adj_matrix: (N, N) 空间邻接矩阵
    """
    print("="*60)
    print("Training EP-STD Diffusion Model")
    print("="*60)

    # 冻结环境编码器
    env_encoder.eval()
    for param in env_encoder.parameters():
        param.requires_grad = False

    # 加载多图结构（如果存在）
    graph_dict = {}
    for graph_name in ['adj_adaptive', 'adj_distance', 'adj_crime_dynamic_gaussian']:
        graph_path = f'data/processed/{graph_name}.npy'
        if os.path.exists(graph_path):
            graph_dict[graph_name] = torch.tensor(np.load(graph_path), dtype=torch.float32).to(device)
            print(f"Loaded {graph_name}: shape {graph_dict[graph_name].shape}")

    # 创建模型（多图融合）
    num_nodes = Y_train.shape[1]
    num_graphs = len(graph_dict) if graph_dict else 1
    model = ConditionalRiskDiffusion(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=4,
        time_dim=64,
        env_dim=64,
        num_prototypes=prototype_library.n_prototypes,
        num_graphs=num_graphs
    ).to(device)

    scheduler = DiffusionScheduler(num_timesteps=100)  # 减少步数加速采样

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 数据归一化：计算Y_train的统计信息用于反归一化
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-6  # 避免除零
    print(f"Y_train stats: mean={y_mean:.4f}, std={y_std:.4f}, max={Y_train.max():.4f}")

    # 准备原型标签
    env_embeddings = np.load('data/processed/env_embeddings.npy')
    prototype_labels = np.load('data/processed/prototype_labels.npy')
    prototype_labels_tensor = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    # 加载多图结构（如果存在）
    graph_dict = {}
    for graph_name in ['adj_adaptive', 'adj_distance', 'adj_crime_dynamic_gaussian']:
        graph_path = f'data/processed/{graph_name}.npy'
        if os.path.exists(graph_path):
            graph_dict[graph_name] = torch.tensor(np.load(graph_path), dtype=torch.float32).to(device)
            print(f"Loaded {graph_name}: shape {graph_dict[graph_name].shape}")

    # 训练循环
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_noise_loss = 0
        total_zinb_loss = 0

        # 随机采样批次
        n_samples = len(Y_train)
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            B = len(batch_idx)

            # 获取数据并归一化
            y_batch = torch.tensor(Y_train[batch_idx], dtype=torch.float32).to(device)
            y_batch_norm = (y_batch - y_mean) / y_std  # 归一化到标准正态

            # 编码环境
            with torch.no_grad():
                x_static = X_train[batch_idx, :, :24]
                x_static_tensor = torch.tensor(x_static, dtype=torch.float32).to(device)
                # 将3D reshape为2D用于编码器
                B_actual, N, F = x_static_tensor.shape
                x_static_flat = x_static_tensor.view(B_actual * N, F)
                env_emb_flat = env_encoder(x_static_flat)  # (B*N, env_dim)
                env_emb = env_emb_flat.view(B_actual, N, -1)  # (B, N, env_dim)

            # 采样时间步
            t = scheduler.sample_timesteps(B, device)

            # 加噪（使用归一化后的数据）
            noise = torch.randn_like(y_batch_norm)
            x_t = scheduler.add_noise(y_batch_norm, t, noise)

            # 扩展原型标签到batch
            proto_ids = prototype_labels_tensor.unsqueeze(0).expand(B, -1)

            # 前向传播（使用多图）
            adj_list = list(graph_dict.values()) if graph_dict else None
            noise_pred, pi = model(x_t, t, env_emb, proto_ids, adj_list)

            # 噪声预测损失
            noise_loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # ZINB损失：区分零犯罪和正犯罪（降低权重避免过度抑制）
            zero_mask = (y_batch == 0).float()
            # 使用类别权重平衡零/非零样本
            zero_ratio = zero_mask.mean()
            pos_weight = torch.tensor([zero_ratio / (1 - zero_ratio + 1e-6)]).to(device)
            zinb_loss = torch.nn.functional.binary_cross_entropy(pi, zero_mask, weight=(1 - zero_mask) * pos_weight + zero_mask)

            # 总损失 - 大幅降低ZINB权重，让扩散模型主导训练
            loss = noise_loss + 0.01 * zinb_loss

            # 添加稀疏性正则化：鼓励模型在非零区域有响应
            # 计算预测的标准差，鼓励多样性
            diversity_loss = -0.001 * torch.std(noise_pred)
            loss = loss + diversity_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_noise_loss += noise_loss.item()
            total_zinb_loss += zinb_loss.item()

        avg_loss = total_loss / (n_samples // batch_size + 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Noise: {total_noise_loss/(n_samples//batch_size+1):.4f} | "
                  f"ZINB: {total_zinb_loss/(n_samples//batch_size+1):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoints/epstd_diffusion_best.pt')

    print(f"\nTraining completed. Best loss: {best_loss:.4f}")
    print("Model saved to checkpoints/epstd_diffusion_best.pt")

    return model, scheduler, y_mean, y_std


def evaluate_epstd(model, scheduler, env_encoder, prototype_library, X_test, Y_test,
                   y_mean=0.0, y_std=1.0, device='cuda'):
    """评估EP-STD模型"""
    print("\n" + "="*60)
    print("Evaluating EP-STD Model")
    print("="*60)

    predictor = EPSTDPredictor(
        env_encoder, prototype_library, model, scheduler, device,
        y_mean=y_mean, y_std=y_std
    )

    # 取最后一个样本进行预测
    static_features = X_test[-1, :, :24]

    print("Generating predictions (this may take a while)...")
    risk_mean, risk_std, pi = predictor.predict(static_features, num_samples=10)

    y_true = Y_test[-1]

    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from train_stgcn_trans import calculate_advanced_metrics

    mae = mean_absolute_error(y_true, risk_mean)
    rmse = np.sqrt(mean_squared_error(y_true, risk_mean))

    # 热点预测评估 - 需要2维输入 (batch, N)
    risk_mean_2d = risk_mean.reshape(1, -1)
    y_true_2d = y_true.reshape(1, -1)
    metrics = calculate_advanced_metrics(y_true_2d, risk_mean_2d)

    print(f"\nEvaluation Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Recall@Top10%: {metrics.get('recall_top10', 0):.4f}")
    print(f"  PAI: {metrics.get('pai_top10', 0):.4f}")
    print(f"  PEI: {metrics.get('pei_top10', 0):.4f}")

    # 零膨胀检测准确率
    zero_pred = (pi > 0.5).astype(float)
    zero_true = (y_true == 0).astype(float)
    zero_acc = (zero_pred == zero_true).mean()
    print(f"  Zero-inflation Accuracy: {zero_acc:.4f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'metrics': metrics,
        'predictions': risk_mean,
        'uncertainty': risk_std,
        'zero_prob': pi
    }


def main():
    """主流程"""
    print("="*60)
    print("EP-STD Stage 3: Conditional Diffusion Model")
    print("="*60)

    # 加载数据
    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    # 加载环境编码器
    from epstd_stage1 import EnvironmentEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)

    checkpoint_path = 'checkpoints/env_encoder_best.pt'
    if os.path.exists(checkpoint_path):
        env_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded encoder from {checkpoint_path}")

    env_encoder.eval()

    # 加载原型库
    import pickle
    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']
    prototype_library.prototype_risk_dists = proto_data['prototype_risk_dists']

    print(f"Loaded prototype library with {proto_data['n_prototypes']} prototypes")

    # 划分训练/测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # 训练扩散模型
    model, scheduler, y_mean, y_std = train_diffusion_model(
        env_encoder, prototype_library, X_train, Y_train,
        epochs=100, batch_size=16, device=device
    )

    # 加载最佳模型进行评估
    checkpoint = torch.load('checkpoints/epstd_diffusion_best.pt')
    model.load_state_dict(checkpoint['model'])

    # 评估
    results = evaluate_epstd(
        model, scheduler, env_encoder, prototype_library, X_test, Y_test,
        y_mean=y_mean, y_std=y_std, device=device
    )

    # 保存预测结果
    np.save(f'{data_dir}/epstd_predictions.npy', results['predictions'])
    np.save(f'{data_dir}/epstd_uncertainty.npy', results['uncertainty'])
    np.save(f'{data_dir}/epstd_zero_prob.npy', results['zero_prob'])

    print("\n" + "="*60)
    print("Stage 3 completed!")
    print("EP-STD model trained and evaluated.")
    print("="*60)


if __name__ == "__main__":
    main()
