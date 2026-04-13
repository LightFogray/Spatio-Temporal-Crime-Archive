"""
EP-STD Stage 2: 原型学习与生成模型基础
目标：在环境嵌入上建立风险原型，为扩散模型做准备
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


class PrototypeLibrary:
    """
    风险原型库：环境嵌入 -> 风险原型
    使用纯PyTorch实现K-Means聚类，避免sklearn依赖
    """

    def __init__(self, n_prototypes=10):
        self.n_prototypes = n_prototypes
        self.prototypes = None  # (K, D) 原型中心
        self.prototype_risks = None  # (K,) 每个原型的平均风险
        self.prototype_risk_dists = None  # 每个原型的风险分布参数
        self.labels_ = None  # 聚类标签

    def _kmeans_torch(self, X, max_iter=100, tol=1e-4):
        """
        PyTorch实现的K-Means聚类

        Args:
            X: (N, D) numpy array
        Returns:
            labels: (N,) 聚类标签
            centers: (K, D) 聚类中心
        """
        N, D = X.shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 转换为tensor
        X_torch = torch.tensor(X, dtype=torch.float32).to(device)

        # K-Means++初始化
        np.random.seed(42)
        centers = torch.zeros(self.n_prototypes, D, device=device)

        # 随机选择第一个中心
        first_idx = np.random.randint(N)
        centers[0] = X_torch[first_idx]

        for k in range(1, self.n_prototypes):
            # 计算每个点到最近中心的距离
            dists = torch.cdist(X_torch, centers[:k])  # (N, k)
            min_dists = dists.min(dim=1)[0]  # (N,)

            # 按概率选择下一个中心
            probs = min_dists / min_dists.sum()
            next_idx = np.random.choice(N, p=probs.cpu().numpy())
            centers[k] = X_torch[next_idx]

        # K-Means迭代
        for iteration in range(max_iter):
            # 分配步骤：计算每个点到所有中心的距离
            dists = torch.cdist(X_torch, centers)  # (N, K)
            labels = dists.argmin(dim=1)  # (N,)

            # 更新步骤：重新计算中心
            new_centers = torch.zeros_like(centers)
            for k in range(self.n_prototypes):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centers[k] = X_torch[mask].mean(dim=0)
                else:
                    # 空簇：重新随机初始化
                    new_centers[k] = X_torch[np.random.randint(N)]

            # 检查收敛
            center_shift = torch.norm(new_centers - centers, dim=1).max()
            centers = new_centers

            if center_shift < tol:
                print(f"  K-Means converged at iteration {iteration+1}")
                break

        return labels.cpu().numpy(), centers.cpu().numpy()

    def fit(self, env_embeddings, risk_labels):
        """
        在环境嵌入上聚类，建立原型

        Args:
            env_embeddings: (N, D) 环境嵌入
            risk_labels: (N,) 风险标签
        """
        print(f"Building {self.n_prototypes} prototypes from {len(env_embeddings)} grids...")

        # K-Means聚类 (PyTorch实现)
        labels, self.prototypes = self._kmeans_torch(env_embeddings)
        self.labels_ = labels

        # 计算每个原型的风险统计
        self.prototype_risks = []
        self.prototype_risk_dists = []
        self.prototype_members = []

        for k in range(self.n_prototypes):
            mask = (labels == k)
            member_risks = risk_labels[mask]

            self.prototype_members.append(np.where(mask)[0])

            if len(member_risks) > 0:
                mean_risk = member_risks.mean()
                std_risk = member_risks.std()
            else:
                mean_risk = 0.0
                std_risk = 0.1

            self.prototype_risks.append(mean_risk)
            self.prototype_risk_dists.append({'mean': mean_risk, 'std': std_risk})

        self.prototype_risks = np.array(self.prototype_risks)

        # 打印原型信息
        print("\nPrototype Summary:")
        for k in range(self.n_prototypes):
            n_members = len(self.prototype_members[k])
            risk_info = self.prototype_risk_dists[k]
            print(f"  Prototype {k:2d}: n={n_members:4d}, "
                  f"risk={risk_info['mean']:.3f}±{risk_info['std']:.3f}")

        return labels

    def query(self, env_embedding):
        """
        查询最近原型

        Returns:
            proto_id: 最近原型ID
            distance: 距离
            risk_prior: 该原型的风险先验
        """
        # 计算到所有原型的距离
        distances = np.linalg.norm(self.prototypes - env_embedding, axis=1)
        proto_id = np.argmin(distances)

        return {
            'proto_id': proto_id,
            'distance': distances[proto_id],
            'risk_prior': self.prototype_risks[proto_id],
            'risk_dist': self.prototype_risk_dists[proto_id]
        }

    def cold_start_predict(self, static_features, env_encoder):
        """
        冷启动预测：无历史犯罪时，用环境找原型预测风险
        """
        # 编码环境
        with torch.no_grad():
            device = next(env_encoder.parameters()).device
            x = torch.tensor(static_features, dtype=torch.float32).to(device)
            env_emb = env_encoder(x).cpu().numpy()

        # 找最近原型
        predictions = []
        for emb in env_emb:
            result = self.query(emb)
            predictions.append({
                'risk_mean': result['risk_prior'],
                'risk_std': result['risk_dist']['std'],
                'confidence': 1.0 / (1.0 + result['distance'])  # 距离越近置信度越高
            })

        return predictions


def visualize_prototypes(env_embeddings, risk_labels, prototype_labels,
                         save_path='prototypes_visualization.png'):
    """可视化原型分布"""
    # PCA降维到2D (使用NumPy实现)
    # 中心化
    mean = env_embeddings.mean(axis=0)
    centered = env_embeddings - mean

    # SVD分解
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # 计算解释方差比例
    total_variance = np.sum(S**2)
    explained_variance_ratio = (S**2) / total_variance

    # 取前2个主成分
    embedded = U[:, :2] * S[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按原型聚类着色
    scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1],
                               c=prototype_labels, cmap='tab10',
                               s=30, alpha=0.6)
    axes[0].set_title('Environment Prototypes')
    axes[0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    plt.colorbar(scatter1, ax=axes[0], label='Prototype ID')

    # 右图：按风险值着色
    scatter2 = axes[1].scatter(embedded[:, 0], embedded[:, 1],
                               c=risk_labels, cmap='YlOrRd',
                               s=30, alpha=0.6)
    axes[1].set_title('Risk Distribution')
    axes[1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    plt.colorbar(scatter2, ax=axes[1], label='Risk Level')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPrototype visualization saved to {save_path}")


def test_cold_start_prediction(prototype_lib, env_embeddings, risk_labels,
                                static_features, env_encoder, test_ratio=0.2):
    """
    测试冷启动预测能力
    模拟：随机隐藏一部分网格的历史，只用环境预测
    """
    print("\n" + "="*60)
    print("Cold-start Prediction Test")
    print("="*60)

    n_test = int(len(env_embeddings) * test_ratio)
    test_indices = np.random.choice(len(env_embeddings), n_test, replace=False)

    # 冷启动预测（只用环境，不看历史风险）
    test_env = static_features[test_indices]
    predictions = prototype_lib.cold_start_predict(test_env, env_encoder)

    pred_risks = [p['risk_mean'] for p in predictions]
    true_risks = risk_labels[test_indices]

    # 计算误差
    mae = np.mean(np.abs(np.array(pred_risks) - true_risks))
    correlation = np.corrcoef(pred_risks, true_risks)[0, 1]

    print(f"Test samples: {n_test}")
    print(f"MAE: {mae:.4f}")
    print(f"Correlation: {correlation:.4f}")

    if correlation > 0.5:
        print("[PASS] Cold-start prediction is effective")
    else:
        print("[WARN] Cold-start prediction needs improvement")

    return {'mae': mae, 'correlation': correlation, 'predictions': pred_risks, 'true': true_risks}


def main():
    """主流程"""
    print("="*60)
    print("EP-STD Stage 2: Prototype Learning")
    print("="*60)

    # 加载阶段1的结果
    data_dir = 'data/processed'
    env_embeddings = np.load(f'{data_dir}/env_embeddings.npy')
    static_features = np.load(f'{data_dir}/X.npy')[-1, :, :24]
    risk_labels = np.load(f'{data_dir}/Y.npy')[-1, :]

    print(f"Loaded {len(env_embeddings)} environment embeddings")

    # 加载阶段1的编码器
    from epstd_stage1 import EnvironmentEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)

    checkpoint_path = 'checkpoints/env_encoder_best.pt'
    if os.path.exists(checkpoint_path):
        env_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded encoder from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")

    env_encoder.eval()

    # 构建原型库
    n_prototypes = 10  # 可调整：尝试5, 10, 15, 20
    prototype_lib = PrototypeLibrary(n_prototypes=n_prototypes)
    proto_labels = prototype_lib.fit(env_embeddings, risk_labels)

    # 可视化
    visualize_prototypes(env_embeddings, risk_labels, proto_labels)

    # 测试冷启动预测
    test_results = test_cold_start_prediction(
        prototype_lib, env_embeddings, risk_labels,
        static_features, env_encoder, test_ratio=0.2
    )

    # 保存原型库
    import pickle
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/prototype_library.pkl', 'wb') as f:
        pickle.dump({
            'prototypes': prototype_lib.prototypes,
            'prototype_risks': prototype_lib.prototype_risks,
            'prototype_risk_dists': prototype_lib.prototype_risk_dists,
            'n_prototypes': prototype_lib.n_prototypes
        }, f)
    print(f"\nPrototype library saved to checkpoints/prototype_library.pkl")

    # 保存用于阶段3的数据
    np.save(f'{data_dir}/prototype_labels.npy', proto_labels)
    np.save(f'{data_dir}/prototype_centers.npy', prototype_lib.prototypes)
    print(f"Prototype data saved to {data_dir}/")

    print("\n" + "="*60)
    print("Stage 2 completed!")
    print(f"Next: Stage 3 (Diffusion Model with {n_prototypes} prototypes)")
    print("="*60)


if __name__ == "__main__":
    main()
