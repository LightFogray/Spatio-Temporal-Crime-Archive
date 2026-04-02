"""
消融实验专用脚本
自动运行各模型变体并汇总消融结果
用于验证每个组件的贡献
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.train_stgcn_trans import (
    SpatioTemporalTransformer, CrimeDataset,
    zinb_loss, calculate_metrics
)
from torch.utils.data import DataLoader


# ================================
# 配置参数
# ================================
class AblationConfig:
    """消融实验配置"""
    DATA_DIR = "data/processed"
    CHECKPOINT_DIR = "checkpoints/ablation"
    RESULT_DIR = "experiments/results"

    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-3
    PATIENCE = 15

    HIDDEN_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.1

    SEED = 42  # 消融实验使用固定种子
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(AblationConfig.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AblationConfig.RESULT_DIR, exist_ok=True)


# ================================
# 消融变体定义
# ================================

ABLATION_VARIANTS = {
    # 主消融实验
    'Full_Model': {
        'description': '完整模型',
        'use_semantic_gate': True,
        'use_near_repeat': True,
        'semantic_mode': 'gate',  # gate, concat, none
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'w/o_Semantic': {
        'description': '移除LLM语义嵌入',
        'use_semantic_gate': False,
        'use_near_repeat': True,
        'semantic_mode': 'none',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'w/o_Semantic_Gate': {
        'description': '语义直接拼接（无门控）',
        'use_semantic_gate': False,
        'use_near_repeat': True,
        'semantic_mode': 'concat',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'w/o_Near_Repeat': {
        'description': '移除近重复效应模块',
        'use_semantic_gate': True,
        'use_near_repeat': False,
        'semantic_mode': 'gate',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'w/o_Hypergraph': {
        'description': '移除超图注意力',
        'use_semantic_gate': True,
        'use_near_repeat': True,
        'semantic_mode': 'gate',
        'use_hypergraph': False,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'w/o_Cross_Fusion': {
        'description': '静态/动态特征直接相加',
        'use_semantic_gate': True,
        'use_near_repeat': True,
        'semantic_mode': 'gate',
        'use_hypergraph': True,
        'use_cross_fusion': False,
        'loss_type': 'zinb'
    },

    # 组件替换消融
    'Gating_to_Attention': {
        'description': '门控替换为注意力融合',
        'use_semantic_gate': False,  # 使用注意力
        'use_near_repeat': True,
        'semantic_mode': 'attention',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb'
    },

    'NR_Fixed_Params': {
        'description': '近重复效应使用固定参数',
        'use_semantic_gate': True,
        'use_near_repeat': True,
        'semantic_mode': 'gate',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'zinb',
        'nr_fixed': True  # 固定参数
    },

    'Loss_MSE': {
        'description': 'ZINB替换为MSE损失',
        'use_semantic_gate': True,
        'use_near_repeat': True,
        'semantic_mode': 'gate',
        'use_hypergraph': True,
        'use_cross_fusion': True,
        'loss_type': 'mse'
    }
}


# ================================
# 消融变体模型
# ================================

class AblationModel(nn.Module):
    """
    消融实验专用模型包装器
    根据配置动态调整模型结构
    """

    def __init__(self, base_model_class, config: dict,
                 static_dim, dynamic_dim, semantic_dim, num_nodes):
        super().__init__()

        self.config = config
        self.loss_type = config.get('loss_type', 'zinb')

        # 创建基础模型
        model_kwargs = {
            'static_dim': static_dim,
            'dynamic_dim': dynamic_dim,
            'semantic_dim': semantic_dim if config['semantic_mode'] != 'none' else 0,
            'hidden_dim': AblationConfig.HIDDEN_DIM,
            'num_heads': AblationConfig.NUM_HEADS,
            'dropout': AblationConfig.DROPOUT,
            'num_nodes': num_nodes,
            'use_semantic_gate': config['use_semantic_gate'] and config['semantic_mode'] == 'gate',
            'use_near_repeat': config['use_near_repeat']
        }

        self.base_model = base_model_class(**model_kwargs)

        # 特殊组件处理
        if config['semantic_mode'] == 'attention':
            # 替换门控为注意力
            from src.train_stgcn_trans import CrossAttentionFusion
            self.semantic_fusion = CrossAttentionFusion(
                AblationConfig.HIDDEN_DIM,
                AblationConfig.NUM_HEADS,
                AblationConfig.DROPOUT
            )

        if not config['use_cross_fusion']:
            # 移除交叉融合，使用简单相加
            self.cross_fusion = None
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        if not config['use_hypergraph']:
            # 禁用超图注意力
            self.base_model.hypergraph_attn = lambda x, H, **kwargs: (x, None)

    def forward(self, X, A_spatial, A_distance, A_crime, A_hypergraph,
                OD=None, semantic_embed=None, crime_history=None, return_attention=False):
        """前向传播"""

        # 调用基础模型
        outputs = self.base_model(
            X, A_spatial, A_distance, A_crime, A_hypergraph,
            OD, semantic_embed, return_attention, crime_history
        )

        return outputs


# ================================
# 训练函数
# ================================

def train_ablation_variant(variant_name: str, variant_config: dict,
                           train_loader, val_loader, test_loader,
                           A_spatial, A_distance, A_hypergraph,
                           static_dim, dynamic_dim, semantic_dim, num_nodes,
                           semantic_embed=None):
    """
    训练单个消融变体

    Returns:
        results: 包含各项评估指标的字典
    """
    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"Description: {variant_config['description']}")
    print(f"{'='*60}")

    # 设置随机种子
    torch.manual_seed(AblationConfig.SEED)
    np.random.seed(AblationConfig.SEED)

    # 创建模型
    model = AblationModel(
        SpatioTemporalTransformer,
        variant_config,
        static_dim, dynamic_dim, semantic_dim, num_nodes
    ).to(AblationConfig.DEVICE)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AblationConfig.LR,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 损失函数
    def compute_loss(Y_true, pi, mu, theta):
        if variant_config.get('loss_type') == 'mse':
            pred = (1 - pi) * mu
            return torch.nn.functional.mse_loss(pred, Y_true)
        else:
            return zinb_loss(Y_true, pi, mu, theta)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(AblationConfig.EPOCHS):
        # 训练
        model.train()
        train_losses = []

        for batch in train_loader:
            X_batch, A_crime_batch, OD_batch, Y_batch = batch
            X_batch = X_batch.to(AblationConfig.DEVICE)
            Y_batch = Y_batch.to(AblationConfig.DEVICE)
            A_crime_batch = A_crime_batch.to(AblationConfig.DEVICE)

            optimizer.zero_grad()

            # 提取犯罪历史
            crime_history = X_batch[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            # 前向传播
            outputs = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph,
                OD_batch, semantic_embed, crime_history
            )

            pi, mu, theta = outputs[0], outputs[1], outputs[2]

            loss = compute_loss(Y_batch, pi, mu, theta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                X_batch, A_crime_batch, OD_batch, Y_batch = batch
                X_batch = X_batch.to(AblationConfig.DEVICE)
                Y_batch = Y_batch.to(AblationConfig.DEVICE)
                A_crime_batch = A_crime_batch.to(AblationConfig.DEVICE)

                crime_history = X_batch[:, :, :, -7:]
                crime_history = crime_history[:, :, :, 0]

                outputs = model(
                    X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph,
                    OD_batch, semantic_embed, crime_history
                )

                pi, mu, theta = outputs[0], outputs[1], outputs[2]
                loss = compute_loss(Y_batch, pi, mu, theta)
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{AblationConfig.EPOCHS} | "
                  f"Train: {mean_train:.4f} | Val: {mean_val:.4f}")

        # 早停
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(model.state_dict(),
                      f"{AblationConfig.CHECKPOINT_DIR}/{variant_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= AblationConfig.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # 测试评估
    model.load_state_dict(torch.load(f"{AblationConfig.CHECKPOINT_DIR}/{variant_name}_best.pt"))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            X_batch, A_crime_batch, OD_batch, Y_batch = batch
            X_batch = X_batch.to(AblationConfig.DEVICE)
            Y_batch = Y_batch.to(AblationConfig.DEVICE)
            A_crime_batch = A_crime_batch.to(AblationConfig.DEVICE)

            crime_history = X_batch[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            outputs = model(
                X_batch, A_spatial, A_distance, A_crime_batch, A_hypergraph,
                OD_batch, semantic_embed, crime_history
            )

            pi, mu, theta = outputs[0], outputs[1], outputs[2]
            pred = torch.clamp((1 - pi) * mu, min=0)

            preds.append(pred.cpu().numpy())
            targets.append(Y_batch.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(targets)

    results = calculate_metrics(y_true, y_pred, k_percent=0.1)

    print(f"\nResults for {variant_name}:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results


# ================================
# 主流程
# ================================

def load_and_prepare_data():
    """加载并准备数据"""
    print("Loading data...")

    X = np.load(f"{AblationConfig.DATA_DIR}/X.npy")
    Y = np.load(f"{AblationConfig.DATA_DIR}/Y.npy")
    OD = np.load(f"{AblationConfig.DATA_DIR}/dynamic_od_flow_1246.npy")
    OD = np.log1p(OD)

    A_spatial = np.load(f"{AblationConfig.DATA_DIR}/adj_adaptive.npy")
    A_distance = np.load(f"{AblationConfig.DATA_DIR}/adj_distance.npy")
    A_crime = np.load(f"{AblationConfig.DATA_DIR}/adj_crime_dynamic_gaussian.npy")
    A_hypergraph = np.load(f"{AblationConfig.DATA_DIR}/adj_hypergraph.npy")

    semantic_embed = None
    semantic_path = f"{AblationConfig.DATA_DIR}/semantic_embedding.npy"
    if os.path.exists(semantic_path):
        semantic_embed = np.load(semantic_path)

    # 构建窗口
    window = 30
    crime_lag = 7
    offset = window - crime_lag

    X_window = []
    Y_window = []
    A_crime_window = []
    OD_window = []

    for i in range(len(X) - offset):
        X_window.append(X[i:i+offset])
        Y_window.append(Y[i+offset])
        if i + offset < len(A_crime):
            A_crime_window.append(A_crime[i+offset])
        if i + offset < len(OD):
            OD_window.append(OD[i+offset])

    X_window = np.stack(X_window, axis=0).astype(np.float32)
    Y_window = np.stack(Y_window, axis=0).astype(np.float32)
    A_crime_window = np.array(A_crime_window).astype(np.float32)
    OD_window = np.array(OD_window).astype(np.float32)

    # 划分数据集
    n_samples = X_window.shape[0]
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    splits = {
        'X_train': X_window[:train_end],
        'Y_train': Y_window[:train_end],
        'A_crime_train': A_crime_window[:train_end],
        'OD_train': OD_window[:train_end],
        'X_val': X_window[train_end:val_end],
        'Y_val': Y_window[train_end:val_end],
        'A_crime_val': A_crime_window[train_end:val_end],
        'OD_val': OD_window[train_end:val_end],
        'X_test': X_window[val_end:],
        'Y_test': Y_window[val_end:],
        'A_crime_test': A_crime_window[val_end:],
        'OD_test': OD_window[val_end:]
    }

    return splits, A_spatial, A_distance, A_hypergraph, semantic_embed


def generate_ablation_report(results: Dict[str, Dict], save_path: str):
    """生成消融实验报告"""

    # 创建DataFrame
    df = pd.DataFrame(results).T

    # 计算相对于Full Model的性能下降
    if 'Full_Model' in df.index:
        full_metrics = df.loc['Full_Model']

        df['PAI_drop'] = df.apply(
            lambda row: ((full_metrics['PAI'] - row['PAI']) / full_metrics['PAI'] * 100)
            if row.name != 'Full_Model' else 0, axis=1
        )

        df['RMSE_increase'] = df.apply(
            lambda row: ((row['RMSE'] - full_metrics['RMSE']) / full_metrics['RMSE'] * 100)
            if row.name != 'Full_Model' else 0, axis=1
        )

    # 保存CSV
    df.to_csv(save_path.replace('.txt', '.csv'))

    # 生成Markdown报告
    with open(save_path.replace('.txt', '.md'), 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write("## Main Ablation Experiments\n\n")
        f.write(df.to_markdown())
        f.write("\n\n")

        # 瀑布图数据
        f.write("## Performance Degradation Analysis\n\n")
        f.write("Relative to Full Model:\n\n")

        for variant in ABLATION_VARIANTS.keys():
            if variant != 'Full_Model' and variant in df.index:
                pai_drop = df.loc[variant, 'PAI_drop']
                f.write(f"- **{variant}**: PAI drops by {pai_drop:.2f}%\n")

    print(f"\nAblation report saved to:")
    print(f"  - CSV: {save_path.replace('.txt', '.csv')}")
    print(f"  - Markdown: {save_path.replace('.txt', '.md')}")

    return df


def main():
    """主函数"""
    print("=" * 80)
    print("Ablation Study Script")
    print("=" * 80)

    # 加载数据
    splits, A_spatial, A_distance, A_hypergraph, semantic_embed = load_and_prepare_data()

    # 获取维度
    static_dim = 24
    dynamic_dim = splits['X_train'].shape[3] - static_dim
    semantic_dim = semantic_embed.shape[1] if semantic_embed is not None else 0
    num_nodes = splits['X_train'].shape[2]

    print(f"Static dim: {static_dim}, Dynamic dim: {dynamic_dim}")
    print(f"Semantic dim: {semantic_dim}, Num nodes: {num_nodes}")

    # 创建DataLoader
    train_dataset = CrimeDataset(splits['X_train'], splits['Y_train'],
                                  splits['A_crime_train'], splits['OD_train'])
    val_dataset = CrimeDataset(splits['X_val'], splits['Y_val'],
                                splits['A_crime_val'], splits['OD_val'])
    test_dataset = CrimeDataset(splits['X_test'], splits['Y_test'],
                                 splits['A_crime_test'], splits['OD_test'])

    train_loader = DataLoader(train_dataset, batch_size=AblationConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=AblationConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=AblationConfig.BATCH_SIZE, shuffle=False)

    # 转换图为tensor
    A_spatial = torch.tensor(A_spatial, dtype=torch.float32).to(AblationConfig.DEVICE)
    A_distance = torch.tensor(A_distance, dtype=torch.float32).to(AblationConfig.DEVICE)
    A_hypergraph = torch.tensor(A_hypergraph, dtype=torch.float32).to(AblationConfig.DEVICE)

    semantic_tensor = None
    if semantic_embed is not None:
        semantic_tensor = torch.tensor(semantic_embed, dtype=torch.float32).to(AblationConfig.DEVICE)

    # 运行所有消融实验
    all_results = {}

    for variant_name, variant_config in ABLATION_VARIANTS.items():
        results = train_ablation_variant(
            variant_name, variant_config,
            train_loader, val_loader, test_loader,
            A_spatial, A_distance, A_hypergraph,
            static_dim, dynamic_dim, semantic_dim, num_nodes,
            semantic_tensor
        )
        all_results[variant_name] = results

    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{AblationConfig.RESULT_DIR}/ablation_study_{timestamp}.txt"

    df = generate_ablation_report(all_results, save_path)

    # 打印汇总
    print("\n" + "=" * 80)
    print("Ablation Study Summary")
    print("=" * 80)
    print(df.to_string())

    # 保存JSON
    with open(f"{AblationConfig.RESULT_DIR}/ablation_results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Ablation study completed!")
    print(f"Results saved to: {AblationConfig.RESULT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
