"""
风险感知加权数据集
对"环境高风险但历史为0"的样本进行上采样
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler


class RiskAwareSampler(Sampler):
    """
    风险感知采样器：提高冷启动高风险样本的采样概率
    """

    def __init__(self, dataset, env_features, history_crimes,
                 high_risk_percentile=80, zero_history_threshold=0.01,
                 upsample_factor=3.0):
        """
        Args:
            dataset: 原始数据集
            env_features: 环境特征 (N, F)
            history_crimes: 历史犯罪数 (N,)
            high_risk_percentile: 环境风险高的阈值（百分位）
            zero_history_threshold: 历史为0的阈值
            upsample_factor: 上采样倍数
        """
        self.dataset = dataset
        self.upsample_factor = upsample_factor

        # 计算环境风险评分
        env_risk = self._calculate_env_risk(env_features)

        # 找"环境高风险但历史为0"的样本
        high_risk_threshold = np.percentile(env_risk, high_risk_percentile)

        self.cold_start_high_risk_indices = []
        self.normal_indices = []

        for idx in range(len(dataset)):
            grid_idx = idx % len(env_risk)  # 假设数据按网格循环

            if (env_risk[grid_idx] > high_risk_threshold and
                history_crimes[grid_idx] < zero_history_threshold):
                self.cold_start_high_risk_indices.append(idx)
            else:
                self.normal_indices.append(idx)

        print(f"RiskAwareSampler: {len(self.cold_start_high_risk_indices)} cold-start high-risk samples")
        print(f"                  {len(self.normal_indices)} normal samples")
        print(f"                  Upsample factor: {upsample_factor}x")

    def _calculate_env_risk(self, env_features):
        """基于环境特征计算风险评分"""
        # 商业POI密度 + 道路密度 - 照明 - 监控
        if env_features.shape[1] >= 24:
            risk = (
                env_features[:, 0] * 0.3 +      # 商业POI
                env_features[:, 3] * 0.2 +      # 道路密度
                (1 - env_features[:, 8]) * 0.3 + # 低照明
                (1 - env_features[:, 9]) * 0.2   # 低监控
            )
        else:
            risk = env_features.mean(axis=1)
        return risk

    def __iter__(self):
        # 正常样本 + 上采样的冷启动样本
        normal_sample = np.random.permutation(self.normal_indices)
        cold_start_sample = np.random.choice(
            self.cold_start_high_risk_indices,
            size=int(len(self.cold_start_high_risk_indices) * self.upsample_factor),
            replace=True
        )

        # 合并并打乱
        all_indices = np.concatenate([normal_sample, cold_start_sample])
        np.random.shuffle(all_indices)

        return iter(all_indices.tolist())

    def __len__(self):
        return int(len(self.normal_indices) +
                   len(self.cold_start_high_risk_indices) * self.upsample_factor)


class WeightedCrimeDataset(Dataset):
    """
    带样本权重的数据集（可用于加权损失）
    """

    def __init__(self, X, Y, A_crime, OD, env_features=None,
                 use_risk_weighting=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A_crime = torch.tensor(A_crime, dtype=torch.float32) if A_crime is not None else None
        self.OD = torch.tensor(OD, dtype=torch.float32) if OD is not None else None

        # 计算样本权重
        if use_risk_weighting and env_features is not None:
            self.weights = self._calculate_weights(env_features, Y)
        else:
            self.weights = torch.ones(len(self))

    def _calculate_weights(self, env_features, Y):
        """计算样本权重"""
        # 环境风险评分
        env_risk = (
            env_features[:, 0] * 0.3 +
            env_features[:, 3] * 0.2 +
            (1 - env_features[:, 8]) * 0.3 +
            (1 - env_features[:, 9]) * 0.2
        )

        # 历史犯罪（按网格平均）
        if len(Y.shape) > 1:
            history = Y.mean(axis=0) if Y.shape[0] > Y.shape[1] else Y.mean(axis=1)
        else:
            history = Y

        # 高环境风险但低历史 -> 高权重
        weights = np.ones(len(env_features))
        high_env_mask = env_risk > np.percentile(env_risk, 80)
        low_history_mask = history < 0.01

        weights[high_env_mask & low_history_mask] = 2.0  # 双倍权重

        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            'X': self.X[idx],
            'Y': self.Y[idx],
            'weight': self.weights[idx]
        }
        if self.A_crime is not None:
            item['A_crime'] = self.A_crime[idx]
        if self.OD is not None:
            item['OD'] = self.OD[idx]

        return item


# 使用示例
def create_risk_aware_dataloader(X, Y, A_crime, OD, env_features,
                                 batch_size=8, upsample_factor=3.0):
    """
    创建风险感知的数据加载器
    """
    # 计算历史犯罪（按网格平均）
    history_crimes = Y.mean(axis=0) if Y.shape[0] > Y.shape[1] else Y.mean(axis=1)

    # 基础数据集
    base_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(A_crime, dtype=torch.float32) if A_crime is not None else torch.zeros(len(X)),
        torch.tensor(OD, dtype=torch.float32) if OD is not None else torch.zeros(len(X), 4)
    )

    # 风险感知采样器
    sampler = RiskAwareSampler(
        base_dataset,
        env_features,
        history_crimes,
        upsample_factor=upsample_factor
    )

    loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    return loader
