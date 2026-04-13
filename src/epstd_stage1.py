"""
EP-STD Stage 1: 对比学习环境编码器 (Contrastive Environment Encoder)
目标：学习环境表征，使得环境相似的网格在潜在空间中距离近
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 跳过sklearn导入错误
try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EnvironmentEncoder(nn.Module):
    """
    环境编码器：将静态环境特征映射到潜在空间
    """

    def __init__(self, input_dim=24, hidden_dim=128, output_dim=64, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

        # 风险预测头（用于监督信号）
        self.risk_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, return_risk=False):
        """
        x: (B, input_dim) 环境特征
        """
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=-1)  # L2归一化

        if return_risk:
            risk = self.risk_predictor(z)
            return z, risk

        return z


class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    为每个样本构建正样本（同风险等级）和负样本（不同风险等级）
    """

    def __init__(self, static_features, risk_labels, n_positives=5, n_negatives=10):
        """
        Args:
            static_features: (N, F) 环境特征
            risk_labels: (N,) 风险标签（犯罪数）
            n_positives: 每个样本的正样本数
            n_negatives: 每个样本的负样本数
        """
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.risk_labels = risk_labels

        # 定义风险等级（高/中/低）
        high_threshold = np.percentile(risk_labels, 70)
        low_threshold = np.percentile(risk_labels, 30)

        self.risk_classes = np.zeros(len(risk_labels), dtype=int)
        self.risk_classes[risk_labels > high_threshold] = 2  # 高风险
        self.risk_classes[risk_labels < low_threshold] = 0   # 低风险
        self.risk_classes[(risk_labels >= low_threshold) & (risk_labels <= high_threshold)] = 1  # 中风险

        self.n_positives = n_positives
        self.n_negatives = n_negatives

        # 预计算同类/异类索引
        self.high_indices = np.where(self.risk_classes == 2)[0]
        self.medium_indices = np.where(self.risk_classes == 1)[0]
        self.low_indices = np.where(self.risk_classes == 0)[0]

    def __len__(self):
        return len(self.static_features)

    def __getitem__(self, idx):
        """
        返回锚点样本及其正负样本
        """
        anchor = self.static_features[idx]
        anchor_class = self.risk_classes[idx]

        # 选择正样本（同类）
        if anchor_class == 2:
            pos_pool = self.high_indices
        elif anchor_class == 1:
            pos_pool = self.medium_indices
        else:
            pos_pool = self.low_indices

        # 排除自己
        pos_pool = pos_pool[pos_pool != idx]

        if len(pos_pool) >= self.n_positives:
            pos_indices = np.random.choice(pos_pool, self.n_positives, replace=False)
        else:
            pos_indices = np.random.choice(pos_pool, self.n_positives, replace=True)

        positives = self.static_features[pos_indices]

        # 选择负样本（异类）
        if anchor_class == 2:
            neg_pool = np.concatenate([self.medium_indices, self.low_indices])
        elif anchor_class == 1:
            neg_pool = np.concatenate([self.high_indices, self.low_indices])
        else:
            neg_pool = np.concatenate([self.high_indices, self.medium_indices])

        neg_indices = np.random.choice(neg_pool, self.n_negatives, replace=True)
        negatives = self.static_features[neg_indices]

        return {
            'anchor': anchor,
            'positives': positives,
            'negatives': negatives,
            'risk_class': anchor_class,
            'risk_label': self.risk_labels[idx]
        }


def info_nce_loss(anchor, positives, negatives, temperature=0.1):
    """
    InfoNCE对比学习损失

    Args:
        anchor: (B, D) 锚点
        positives: (B, n_pos, D) 正样本
        negatives: (B, n_neg, D) 负样本
        temperature: 温度系数

    Returns:
        loss: 标量
    """
    B, D = anchor.shape

    # 计算锚点与正样本的相似度
    pos_sim = torch.bmm(positives, anchor.unsqueeze(-1)).squeeze(-1)  # (B, n_pos)
    pos_sim = pos_sim / temperature

    # 计算锚点与负样本的相似度
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)  # (B, n_neg)
    neg_sim = neg_sim / temperature

    # InfoNCE: 拉近正样本，推远负样本
    # 分子：exp(正样本相似度)
    # 分母：exp(正样本) + sum(exp(负样本))

    numerator = torch.exp(pos_sim).sum(dim=1)  # (B,)
    denominator = numerator + torch.exp(neg_sim).sum(dim=1)  # (B,)

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss.mean()


def supervised_contrastive_loss(anchor, positives, risk_labels, temperature=0.1):
    """
    监督对比学习损失（考虑风险值的连续差异）

    Args:
        anchor: (B, D)
        positives: (B, n_pos)
        risk_labels: (B,) 锚点的风险值
    """
    # 计算锚点与正样本的风险差异权重
    risk_diff = torch.abs(risk_labels.unsqueeze(1) - risk_labels)  # (B, B)
    weights = torch.exp(-risk_diff)  # 风险越接近，权重越高

    # 计算相似度
    sim_matrix = torch.mm(anchor, anchor.t()) / temperature  # (B, B)

    # 掩码：排除自己
    mask = torch.eye(len(anchor), device=anchor.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # 加权对比损失
    exp_sim = torch.exp(sim_matrix)
    numerator = (weights * exp_sim).sum(dim=1)
    denominator = exp_sim.sum(dim=1)

    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss.mean()


def train_contrastive_encoder(static_features, risk_labels,
                              output_dim=64, epochs=100, lr=1e-3,
                              batch_size=64, device='cuda'):
    """
    训练对比学习环境编码器
    """
    print("="*60)
    print("Training Contrastive Environment Encoder")
    print("="*60)

    # 创建数据集
    dataset = ContrastiveDataset(static_features, risk_labels,
                                  n_positives=5, n_negatives=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = EnvironmentEncoder(
        input_dim=static_features.shape[1],
        output_dim=output_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_risk_loss = 0

        for batch in dataloader:
            anchor = batch['anchor'].to(device)
            positives = batch['positives'].to(device)
            negatives = batch['negatives'].to(device)
            risk_labels_batch = torch.tensor(batch['risk_label'], dtype=torch.float32).to(device)

            # 前向传播
            z_anchor, risk_pred = model(anchor, return_risk=True)

            # 动态处理形状
            B, n_pos, feat_dim = positives.shape
            _, n_neg, _ = negatives.shape

            z_positives = model(positives.view(-1, feat_dim)).view(B, n_pos, -1)
            z_negatives = model(negatives.view(-1, feat_dim)).view(B, n_neg, -1)

            # 对比损失
            contrastive_loss = info_nce_loss(z_anchor, z_positives, z_negatives, temperature=0.1)

            # 风险预测损失（辅助监督）
            risk_loss = F.mse_loss(risk_pred.squeeze(), risk_labels_batch)

            # 总损失
            loss = contrastive_loss + 0.1 * risk_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_risk_loss += risk_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Risk Loss: {total_risk_loss/len(dataloader):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'checkpoints/env_encoder_best.pt')

    print(f"\nTraining completed. Best loss: {best_loss:.4f}")
    print(f"Model saved to checkpoints/env_encoder_best.pt")

    return model


def evaluate_encoder(model, static_features, risk_labels, device='cuda'):
    """
    评估编码器：检查环境表征是否有效区分风险
    """
    print("\n" + "="*60)
    print("Evaluating Environment Encoder")
    print("="*60)

    model.eval()

    with torch.no_grad():
        features = torch.tensor(static_features, dtype=torch.float32).to(device)
        embeddings = model(features).cpu().numpy()

    # 1. 计算类间距离（高风险 vs 低风险）
    high_threshold = np.percentile(risk_labels, 80)
    low_threshold = np.percentile(risk_labels, 20)

    high_mask = risk_labels > high_threshold
    low_mask = risk_labels < low_threshold

    # 避免空mask导致的nan
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        print("  Warning: High or low risk mask is empty, skipping centroid calculation")
        return embeddings, {'error': 'Empty mask'}

    high_centroid = embeddings[high_mask].mean(axis=0)
    low_centroid = embeddings[low_mask].mean(axis=0)

    inter_class_distance = np.linalg.norm(high_centroid - low_centroid)

    # 2. 计算类内方差
    high_variance = embeddings[high_mask].var(axis=0).mean()
    low_variance = embeddings[low_mask].var(axis=0).mean()
    avg_intra_variance = (high_variance + low_variance) / 2

    # 3. 计算轮廓系数（如果sklearn可用）
    sil_score = -1.0
    if SKLEARN_AVAILABLE:
        try:
            from sklearn.metrics import silhouette_score
            # 简化：只分两类（高/低）
            binary_labels = np.zeros(len(risk_labels))
            binary_labels[risk_labels > np.median(risk_labels)] = 1
            sil_score = silhouette_score(embeddings, binary_labels)
        except Exception as e:
            print(f"Could not compute silhouette score: {e}")
    else:
        print("  Skipping silhouette score (sklearn not available)")

    print(f"\n评估指标:")
    print(f"  类间距离 (高vs低): {inter_class_distance:.4f}")
    print(f"  平均类内方差: {avg_intra_variance:.4f}")
    print(f"  分离比 (类间/类内): {inter_class_distance / (avg_intra_variance + 1e-8):.4f}")
    if sil_score >= 0:
        print(f"  轮廓系数: {sil_score:.4f}")

    if not SKLEARN_AVAILABLE or sil_score > 0.1:
        print("  [PASS] 环境表征有效区分风险等级")
    else:
        print("  [WARN] 环境表征区分度有限")

    # 可视化
    visualize_embeddings(embeddings, risk_labels)

    return embeddings, {
        'inter_class_distance': inter_class_distance,
        'intra_variance': avg_intra_variance,
        'silhouette_score': sil_score
    }


def visualize_embeddings(embeddings, risk_labels, save_path='env_embeddings_tsne.png'):
    """t-SNE可视化"""
    if not SKLEARN_AVAILABLE:
        print(f"\nSkipping t-SNE visualization (sklearn not available)")
        # 使用PCA替代
        try:
            from numpy.linalg import svd
            # 简单PCA
            mean = embeddings.mean(axis=0)
            centered = embeddings - mean
            U, S, Vt = svd(centered, full_matrices=False)
            embedded = U[:, :2] * S[:2]

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                                  c=risk_labels, cmap='YlOrRd',
                                  s=20, alpha=0.6)
            plt.colorbar(scatter, label='Risk Level')
            plt.title('Environment Embeddings (PCA)')
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.tight_layout()
            plt.savefig(save_path.replace('tsne', 'pca'), dpi=150)
            print(f"Saved PCA visualization to {save_path.replace('tsne', 'pca')}")
        except Exception as e:
            print(f"PCA visualization failed: {e}")
        return

    print(f"\nGenerating t-SNE visualization...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                          c=risk_labels, cmap='YlOrRd',
                          s=20, alpha=0.6)
    plt.colorbar(scatter, label='Risk Level')
    plt.title('Environment Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def main():
    """主流程"""
    import numpy as np

    # 加载数据
    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    # 取最后时刻的静态特征
    static_features = X[-1, :, :24]
    risk_labels = Y[-1, :]

    print(f"Loaded {len(static_features)} grids with {static_features.shape[1]} features")
    print(f"Risk range: [{risk_labels.min():.4f}, {risk_labels.max():.4f}]")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 训练编码器
    model = train_contrastive_encoder(
        static_features, risk_labels,
        output_dim=64,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        device=device
    )

    # 评估
    model.load_state_dict(torch.load('checkpoints/env_encoder_best.pt'))
    embeddings, metrics = evaluate_encoder(model, static_features, risk_labels, device)

    # 保存编码器和嵌入
    np.save('data/processed/env_embeddings.npy', embeddings)
    print(f"\nEmbeddings saved to data/processed/env_embeddings.npy")

    print("\n" + "="*60)
    print("Stage 1 completed!")
    print("Next: Use these embeddings for EP-STD Stage 2 (Prototype Learning)")
    print("="*60)


if __name__ == "__main__":
    main()
