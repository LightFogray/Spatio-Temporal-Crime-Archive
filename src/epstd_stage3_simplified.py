"""
EP-STD Stage 3: 简化版条件扩散模型
核心原则：扩散模型只负责生成，空间关系由Stage 1和Stage 2处理
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
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
    简化的条件扩散模型 - 无图卷积
    核心思想：空间关系由环境条件和原型ID编码，扩散模型只处理特征级别生成
    """

    def __init__(
        self,
        num_nodes=1246,
        hidden_dim=128,
        num_layers=3,
        time_dim=64,
        env_dim=64,
        num_prototypes=10,
        dropout=0.1
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(time_dim, hidden_dim)  # 修正维度
        )

        # 环境条件编码（每个节点独立处理）
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 原型嵌入
        self.prototype_embedding = nn.Embedding(num_prototypes, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(1, hidden_dim)

        # 融合编码：将时间、环境、原型、噪声风险融合
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 去噪网络：纯MLP，无图卷积
        # 空间关系通过环境条件和原型ID隐式编码
        self.denoising_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # 输出投影：预测噪声
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 零膨胀检测头
        self.zero_inflation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_t, t, env_emb, prototype_ids):
        """
        Args:
            x_t: (B, N) 噪声化的风险值
            t: (B,) 时间步
            env_emb: (B, N, env_dim) 环境嵌入（来自Stage 1，已含空间信息）
            prototype_ids: (B, N) 原型ID（来自Stage 2）
        Returns:
            noise_pred: (B, N) 预测的噪声
            pi: (B, N) 零犯罪概率
        """
        B, N = x_t.shape

        # 时间编码并扩展到每个节点
        t_emb = self.time_mlp(t)  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # 环境编码
        h_env = self.env_encoder(env_emb)  # (B, N, hidden_dim)

        # 原型编码
        h_proto = self.prototype_embedding(prototype_ids)  # (B, N, hidden_dim)

        # 噪声风险编码
        h_noise = self.input_proj(x_t.unsqueeze(-1))  # (B, N, hidden_dim)

        # 融合所有条件
        fusion_input = torch.cat([t_emb, h_env, h_proto, h_noise], dim=-1)  # (B, N, hidden_dim*4)
        h = self.fusion_proj(fusion_input)  # (B, N, hidden_dim)

        # 去噪网络（纯MLP，每个节点独立处理）
        for layer in self.denoising_layers:
            h = layer(h) + h  # 残差连接

        # 预测噪声
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, N)

        # 预测零膨胀概率（基于每个节点的特征）
        pi = self.zero_inflation_head(h).squeeze(-1)  # (B, N)

        return noise_pred, pi


class DiffusionScheduler:
    """简化版扩散调度器"""

    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 预计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise):
        """前向扩散"""
        device = x_0.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)
        sqrt_alpha = sqrt_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alphas[t].view(-1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def denoise_step(self, model, x_t, t, env_emb, prototype_ids):
        """单步去噪（简化版，无图结构）"""
        # 预测噪声
        noise_pred, pi = model(x_t, t, env_emb, prototype_ids)

        # 计算x_0预测
        device = x_t.device
        sqrt_alphas = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod.to(device)

        sqrt_alpha = sqrt_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alphas[t].view(-1, 1)
        pred_x_0 = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        # 简化的去噪（DDPM简化版）
        alpha = self.alphas[t].view(-1, 1).to(device)
        pred_mean = (x_t - (1 - alpha) / sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha)

        # 最后一步不加噪声
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.betas[t].view(-1, 1).to(device)
            pred_mean = pred_mean + torch.sqrt(variance) * noise

        return pred_mean, pred_x_0, pi


class EPSTDPredictor:
    """预测器 - 简化版"""

    def __init__(self, env_encoder, prototype_library, diffusion_model,
                 scheduler, device='cuda', y_mean=0.0, y_std=1.0):
        self.env_encoder = env_encoder
        self.prototype_library = prototype_library
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std

        self.env_encoder.eval()
        self.diffusion_model.eval()

    @torch.no_grad()
    def predict(self, static_features, num_samples=10):
        """生成风险预测"""
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
        env_emb = env_emb.unsqueeze(0)
        prototype_ids = prototype_ids.unsqueeze(0)

        # 多次采样去噪
        all_samples = []
        all_pis = []

        for _ in range(num_samples):
            x_t = torch.randn(1, N).to(self.device)

            # 逐步去噪
            for t in reversed(range(self.scheduler.num_timesteps)):
                t_tensor = torch.full((1,), t, dtype=torch.long).to(self.device)
                x_t, pred_x_0, pi = self.scheduler.denoise_step(
                    self.diffusion_model, x_t, t_tensor,
                    env_emb, prototype_ids
                )
                pred_x_0 = torch.clamp(pred_x_0, min=-3, max=3)  # 限制范围

            all_samples.append(pred_x_0.cpu().numpy().squeeze())
            all_pis.append(pi.cpu().numpy().squeeze())

        # 聚合结果
        risk_mean = np.mean(all_samples, axis=0)
        risk_std = np.std(all_samples, axis=0)
        pi_mean = np.mean(all_pis, axis=0)

        # 反归一化
        risk_mean = risk_mean * self.y_std + self.y_mean
        risk_std = risk_std * self.y_std

        # 零膨胀调整
        risk_mean = np.where(pi_mean > 0.7, 0, np.maximum(risk_mean, 0))

        return risk_mean, risk_std, pi_mean


def train_diffusion_model(
    env_encoder, prototype_library, X_train, Y_train,
    epochs=100, batch_size=32, lr=1e-4, device='cuda'
):
    """训练扩散模型 - 简化版"""
    print("="*60)
    print("Training EP-STD Diffusion Model (Simplified)")
    print("="*60)

    # 冻结环境编码器
    env_encoder.eval()
    for param in env_encoder.parameters():
        param.requires_grad = False

    # 数据归一化
    y_mean = Y_train.mean()
    y_std = Y_train.std() + 1e-6
    print(f"Y_train stats: mean={y_mean:.4f}, std={y_std:.4f}")

    # 创建模型
    num_nodes = Y_train.shape[1]
    model = ConditionalRiskDiffusion(
        num_nodes=num_nodes,
        hidden_dim=128,
        num_layers=3,  # 减少层数
        time_dim=64,
        env_dim=64,
        num_prototypes=prototype_library.n_prototypes
    ).to(device)

    scheduler = DiffusionScheduler(num_timesteps=100)  # 减少到100步

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 准备原型标签
    prototype_labels = np.load('data/processed/prototype_labels.npy')
    prototype_labels_tensor = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    # 训练循环
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_noise_loss = 0

        n_samples = len(Y_train)
        indices = np.random.permutation(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            B = len(batch_idx)

            # 获取数据并归一化
            y_batch = torch.tensor(Y_train[batch_idx], dtype=torch.float32).to(device)
            y_batch_norm = (y_batch - y_mean) / y_std

            # 编码环境
            with torch.no_grad():
                x_static = X_train[batch_idx, :, :24]
                x_static_tensor = torch.tensor(x_static, dtype=torch.float32).to(device)
                B_actual, N, F = x_static_tensor.shape
                x_static_flat = x_static_tensor.view(B_actual * N, F)
                env_emb_flat = env_encoder(x_static_flat)
                env_emb = env_emb_flat.view(B_actual, N, -1)

            # 采样时间步并加噪
            t = scheduler.sample_timesteps(B, device)
            noise = torch.randn_like(y_batch_norm)
            x_t = scheduler.add_noise(y_batch_norm, t, noise)

            # 扩展原型标签
            proto_ids = prototype_labels_tensor.unsqueeze(0).expand(B, -1)

            # 前向传播（简化版，无图结构）
            noise_pred, pi = model(x_t, t, env_emb, proto_ids)

            # 损失计算
            noise_loss = F.mse_loss(noise_pred, noise)

            # 简化的ZINB损失
            zero_mask = (y_batch == 0).float()
            zinb_loss = F.binary_cross_entropy(pi, zero_mask)

            # 总损失
            loss = noise_loss + 0.05 * zinb_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_noise_loss += noise_loss.item()

        avg_loss = total_loss / (n_samples // batch_size + 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Noise: {total_noise_loss/(n_samples//batch_size+1):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'y_mean': y_mean,
                'y_std': y_std
            }, 'checkpoints/epstd_diffusion_best.pt')

    print(f"\nTraining completed. Best loss: {best_loss:.4f}")

    return model, scheduler, y_mean, y_std


def evaluate_epstd(model, scheduler, env_encoder, prototype_library,
                   X_test, Y_test, y_mean=0.0, y_std=1.0, device='cuda'):
    """评估模型"""
    print("\n" + "="*60)
    print("Evaluating EP-STD Model")
    print("="*60)

    predictor = EPSTDPredictor(
        env_encoder, prototype_library, model, scheduler,
        device, y_mean, y_std
    )

    static_features = X_test[-1, :, :24]
    y_true = Y_test[-1]

    print("Generating predictions...")
    risk_mean, risk_std, pi = predictor.predict(static_features, num_samples=10)

    # 计算指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, risk_mean)
    rmse = np.sqrt(mean_squared_error(y_true, risk_mean))

    # 热点指标
    from train_stgcn_trans import calculate_advanced_metrics
    risk_mean_2d = risk_mean.reshape(1, -1)
    y_true_2d = y_true.reshape(1, -1)
    metrics = calculate_advanced_metrics(y_true_2d, risk_mean_2d)

    print(f"\nEvaluation Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Recall@Top10%: {metrics.get('recall_top10', 0):.4f}")
    print(f"  PAI: {metrics.get('pai_top10', 0):.4f}")

    zero_acc = ((pi > 0.5) == (y_true == 0)).mean()
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
    print("EP-STD Stage 3: Simplified Diffusion Model")
    print("="*60)

    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    from epstd_stage1 import EnvironmentEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    checkpoint_path = 'checkpoints/env_encoder_best.pt'
    if os.path.exists(checkpoint_path):
        env_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded encoder from {checkpoint_path}")
    env_encoder.eval()

    import pickle
    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    from epstd_stage2 import PrototypeLibrary
    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']
    prototype_library.prototype_risk_dists = proto_data['prototype_risk_dists']

    print(f"Loaded prototype library with {proto_data['n_prototypes']} prototypes")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    model, scheduler, y_mean, y_std = train_diffusion_model(
        env_encoder, prototype_library, X_train, Y_train,
        epochs=100, batch_size=32, device=device
    )

    checkpoint = torch.load('checkpoints/epstd_diffusion_best.pt')
    model.load_state_dict(checkpoint['model'])

    results = evaluate_epstd(
        model, scheduler, env_encoder, prototype_library,
        X_test, Y_test, y_mean, y_std, device
    )

    np.save(f'{data_dir}/epstd_predictions.npy', results['predictions'])
    np.save(f'{data_dir}/epstd_uncertainty.npy', results['uncertainty'])
    np.save(f'{data_dir}/epstd_zero_prob.npy', results['zero_prob'])

    print("\nStage 3 completed!")


if __name__ == "__main__":
    main()
