"""
统一Baseline训练与评估脚本
自动训练所有对比模型并生成结果表格
适用于期刊论文的完整实验流程
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.baselines import (
    HistoricalAverage, RandomForestPredictor,
    ConvLSTM, STGCN, DCRNN, GraphWaveNet,
    calculate_advanced_metrics
)
from src.train_stgcn_trans import (
    SpatioTemporalTransformer, CrimeDataset,
    zinb_loss, train_model, test_model, calculate_advanced_metrics
)
from torch.utils.data import DataLoader


# ================================
# 配置参数
# ================================
class Config:
    """实验配置"""
    # 数据路径
    DATA_DIR = "data/processed"
    CHECKPOINT_DIR = "checkpoints/baselines"
    RESULT_DIR = "experiments/results"

    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15

    # 模型参数
    HIDDEN_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.1

    # 重复实验
    NUM_RUNS = 5  # 用于统计显著性
    SEEDS = [42, 123, 456, 789, 2023]

    # 评估
    K_PERCENT = 0.1  # Top-10% Hit Rate

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.RESULT_DIR, exist_ok=True)


# ================================
# 数据加载工具
# ================================

def load_all_data():
    """加载所有必要数据"""
    print("=" * 60)
    print("Loading data...")

    X = np.load(f"{Config.DATA_DIR}/X.npy")
    Y = np.load(f"{Config.DATA_DIR}/Y.npy")
    OD = np.load(f"{Config.DATA_DIR}/dynamic_od_flow_1246.npy")
    OD = np.log1p(OD)

    A_spatial = np.load(f"{Config.DATA_DIR}/adj_adaptive.npy")
    A_distance = np.load(f"{Config.DATA_DIR}/adj_distance.npy")
    A_crime_dynamic = np.load(f"{Config.DATA_DIR}/adj_crime_dynamic_gaussian.npy")
    A_hypergraph = np.load(f"{Config.DATA_DIR}/adj_hypergraph.npy")

    # 加载语义嵌入（如果有）
    semantic_embed = None
    semantic_path = f"{Config.DATA_DIR}/semantic_embedding.npy"
    if os.path.exists(semantic_path):
        semantic_embed = np.load(semantic_path)
        print(f"Loaded semantic embedding: {semantic_embed.shape}")

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print("=" * 60)

    return {
        'X': X, 'Y': Y, 'OD': OD,
        'A_spatial': A_spatial,
        'A_distance': A_distance,
        'A_crime_dynamic': A_crime_dynamic,
        'A_hypergraph': A_hypergraph,
        'semantic_embed': semantic_embed
    }


def build_window_data(X, Y, A_crime, OD, window=30):
    """构建时间窗口数据"""
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

    return {
        'X': np.stack(X_window, axis=0).astype(np.float32),
        'Y': np.stack(Y_window, axis=0).astype(np.float32),
        'A_crime': np.array(A_crime_window).astype(np.float32) if A_crime_window else None,
        'OD': np.array(OD_window).astype(np.float32) if OD_window else None
    }


def split_dataset(data_dict, train_ratio=0.7, val_ratio=0.15):
    """划分训练/验证/测试集"""
    n_samples = data_dict['X'].shape[0]
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    splits = {}
    for key in ['X', 'Y', 'A_crime', 'OD']:
        if data_dict[key] is not None:
            splits[f'{key}_train'] = data_dict[key][:train_end]
            splits[f'{key}_val'] = data_dict[key][train_end:val_end]
            splits[f'{key}_test'] = data_dict[key][val_end:]

    return splits


# ================================
# 模型训练函数
# ================================

class Trainer:
    """统一训练器"""

    def __init__(self, model_name: str, model, config: Config):
        self.model_name = model_name
        self.model = model
        self.config = config
        self.device = config.DEVICE

    def train_deeplearning(self, train_loader, val_loader,
                           A_spatial, A_distance, A_hypergraph,
                           semantic_embed=None):
        """训练深度学习模型"""
        print(f"\nTraining {self.model_name}...")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.EPOCHS):
            # 训练
            self.model.train()
            train_losses = []

            for batch in train_loader:
                X_batch, A_crime_batch, OD_batch, Y_batch = batch
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                A_crime_batch = A_crime_batch.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                if hasattr(self.model, 'forward'):
                    # 标准前向
                    if isinstance(self.model, (ConvLSTM, STGCN, DCRNN, GraphWaveNet)):
                        pi, mu, theta = self.model(X_batch, A_spatial)
                    else:  # Our model
                        crime_history = X_batch[:, :, :, -7:]
                        crime_history = crime_history[:, :, :, 0]
                        pi, mu, theta, _, _ = self.model(
                            X_batch, A_spatial, A_distance,
                            A_crime_batch, A_hypergraph, OD_batch,
                            semantic_embed=semantic_embed,
                            crime_history=crime_history
                        )

                    loss = zinb_loss(Y_batch, pi, mu, theta)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    X_batch, A_crime_batch, OD_batch, Y_batch = batch
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    A_crime_batch = A_crime_batch.to(self.device)

                    if isinstance(self.model, (ConvLSTM, STGCN, DCRNN, GraphWaveNet)):
                        pi, mu, theta = self.model(X_batch, A_spatial)
                    else:
                        crime_history = X_batch[:, :, :, -7:]
                        crime_history = crime_history[:, :, :, 0]
                        pi, mu, theta, _, _ = self.model(
                            X_batch, A_spatial, A_distance,
                            A_crime_batch, A_hypergraph, OD_batch,
                            semantic_embed=semantic_embed,
                            crime_history=crime_history
                        )

                    loss = zinb_loss(Y_batch, pi, mu, theta)
                    val_losses.append(loss.item())

            mean_train = np.mean(train_losses)
            mean_val = np.mean(val_losses)
            history['train_loss'].append(mean_train)
            history['val_loss'].append(mean_val)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                      f"Train: {mean_train:.4f} | Val: {mean_val:.4f}")

            # 早停
            if mean_val < best_val_loss:
                best_val_loss = mean_val
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(),
                          f"{self.config.CHECKPOINT_DIR}/{self.model_name}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

        return history

    def evaluate(self, test_loader, A_spatial, A_distance, A_hypergraph,
                 semantic_embed=None):
        """评估模型"""
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                X_batch, A_crime_batch, OD_batch, Y_batch = batch
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                A_crime_batch = A_crime_batch.to(self.device)

                if isinstance(self.model, (ConvLSTM, STGCN, DCRNN, GraphWaveNet)):
                    pi, mu, theta = self.model(X_batch, A_spatial)
                else:
                    crime_history = X_batch[:, :, :, -7:]
                    crime_history = crime_history[:, :, :, 0]
                    pi, mu, theta, _, _ = self.model(
                        X_batch, A_spatial, A_distance,
                        A_crime_batch, A_hypergraph, OD_batch,
                        semantic_embed=semantic_embed,
                        crime_history=crime_history
                    )

                pred = torch.clamp((1 - pi) * mu, min=0)
                preds.append(pred.cpu().numpy())
                targets.append(Y_batch.cpu().numpy())

        y_pred = np.vstack(preds)
        y_true = np.vstack(targets)

        return calculate_advanced_metrics(y_true, y_pred, k_percent=self.config.K_PERCENT)


# ================================
# 主实验流程
# ================================

def run_single_experiment(seed: int, all_data: dict) -> Dict:
    """运行单次完整实验"""
    print(f"\n{'='*60}")
    print(f"Running experiment with seed {seed}")
    print(f"{'='*60}")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建窗口数据
    window_data = build_window_data(
        all_data['X'], all_data['Y'],
        all_data['A_crime_dynamic'],
        all_data['OD'],
        window=30
    )

    # 划分数据集
    splits = split_dataset(window_data)

    # 获取维度信息
    T_train, N, F = splits['X_train'].shape
    static_dim = 24  # 假设值，需要根据实际数据调整
    dynamic_dim = F - static_dim
    semantic_dim = all_data['semantic_embed'].shape[1] if all_data['semantic_embed'] is not None else 0

    print(f"Static dim: {static_dim}, Dynamic dim: {dynamic_dim}, Semantic dim: {semantic_dim}")

    # 准备图数据
    A_spatial = torch.tensor(all_data['A_spatial'], dtype=torch.float32).to(Config.DEVICE)
    A_distance = torch.tensor(all_data['A_distance'], dtype=torch.float32).to(Config.DEVICE)
    A_hypergraph = torch.tensor(all_data['A_hypergraph'], dtype=torch.float32).to(Config.DEVICE)

    semantic_embed = None
    if all_data['semantic_embed'] is not None:
        semantic_embed = torch.tensor(all_data['semantic_embed'], dtype=torch.float32).to(Config.DEVICE)

    # 创建DataLoader
    train_dataset = CrimeDataset(splits['X_train'], splits['Y_train'],
                                  splits['A_crime_train'], splits['OD_train'])
    val_dataset = CrimeDataset(splits['X_val'], splits['Y_val'],
                                splits['A_crime_val'], splits['OD_val'])
    test_dataset = CrimeDataset(splits['X_test'], splits['Y_test'],
                                 splits['A_crime_test'], splits['OD_test'])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    results = {}

    # 1. Historical Average
    print("\n[1/8] Historical Average")
    ha = HistoricalAverage(window_size=7)
    ha.fit(splits['X_train'], splits['Y_train'])
    _, mu, _ = ha.predict(splits['X_test'])
    results['HA'] = calculate_advanced_metrics(splits['Y_test'], mu, Config.K_PERCENT)

    # 2. Random Forest
    print("\n[2/8] Random Forest")
    rf = RandomForestPredictor(n_estimators=200)
    rf.fit(splits['X_train'], splits['Y_train'])
    _, mu, _ = rf.predict(splits['X_test'])
    results['RF'] = calculate_advanced_metrics(splits['Y_test'], mu, Config.K_PERCENT)

    # 3. ConvLSTM
    print("\n[3/8] ConvLSTM")
    convlstm = ConvLSTM(input_dim=F, hidden_dim=Config.HIDDEN_DIM,
                        num_layers=2, dropout=Config.DROPOUT).to(Config.DEVICE)
    trainer = Trainer('ConvLSTM', convlstm, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph)
    convlstm.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/ConvLSTM_best.pt"))
    results['ConvLSTM'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                           A_hypergraph)

    # 4. ST-GCN
    print("\n[4/8] ST-GCN")
    stgcn = STGCN(input_dim=F, hidden_dim=Config.HIDDEN_DIM,
                  num_layers=3, dropout=Config.DROPOUT).to(Config.DEVICE)
    trainer = Trainer('STGCN', stgcn, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph)
    stgcn.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/STGCN_best.pt"))
    results['STGCN'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                        A_hypergraph)

    # 5. DCRNN
    print("\n[5/8] DCRNN")
    dcrnn = DCRNN(input_dim=F, hidden_dim=Config.HIDDEN_DIM,
                  num_layers=2, dropout=Config.DROPOUT).to(Config.DEVICE)
    trainer = Trainer('DCRNN', dcrnn, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph)
    dcrnn.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/DCRNN_best.pt"))
    results['DCRNN'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                        A_hypergraph)

    # 6. Graph WaveNet
    print("\n[6/8] Graph WaveNet")
    gwn = GraphWaveNet(input_dim=F, hidden_dim=Config.HIDDEN_DIM,
                       num_nodes=N, dropout=Config.DROPOUT).to(Config.DEVICE)
    trainer = Trainer('GraphWaveNet', gwn, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph)
    gwn.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/GraphWaveNet_best.pt"))
    results['GraphWaveNet'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                                A_hypergraph)

    # 7. ST-Transformer (without semantic) - 消融基线
    print("\n[7/8] ST-Transformer (w/o semantic)")
    stt_no_sem = SpatioTemporalTransformer(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        semantic_dim=0,  # 显式设为0表示无语义
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        num_nodes=N,
        use_semantic_gate=False,  # 关闭自适应融合
        use_near_repeat=True
    ).to(Config.DEVICE)
    trainer = Trainer('STT_NoSemantic', stt_no_sem, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph, None)
    stt_no_sem.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/STT_NoSemantic_best.pt"))
    results['STT_NoSemantic'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                                  A_hypergraph, None)

    # 8. Our Full Model (ACR-ST)
    print("\n[8/8] Our Full Model (ACR-ST with LLM Semantic)")
    our_model = SpatioTemporalTransformer(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        semantic_dim=semantic_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        num_nodes=N,
        use_semantic_gate=True,  # 启用自适应专家融合
        use_near_repeat=True
    ).to(Config.DEVICE)
    trainer = Trainer('ACR-ST', our_model, Config())
    trainer.train_deeplearning(train_loader, val_loader,
                               A_spatial, A_distance, A_hypergraph, semantic_embed)
    our_model.load_state_dict(torch.load(f"{Config.CHECKPOINT_DIR}/ACR-ST_best.pt"))
    results['ACR-ST'] = trainer.evaluate(test_loader, A_spatial, A_distance,
                                       A_hypergraph, semantic_embed)

    return results


def aggregate_results(all_results: List[Dict]) -> pd.DataFrame:
    """聚合多次实验结果，计算均值±标准差"""

    # 按模型聚合
    model_metrics = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        for model_name, metrics in result.items():
            for metric_name, value in metrics.items():
                model_metrics[model_name][metric_name].append(value)

    # 计算统计量
    summary = {}
    for model_name, metrics in model_metrics.items():
        summary[model_name] = {}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[model_name][metric_name] = f"{mean_val:.4f}±{std_val:.4f}"
            summary[model_name][f"{metric_name}_mean"] = mean_val
            summary[model_name][f"{metric_name}_std"] = std_val

    # 创建DataFrame
    df = pd.DataFrame(summary).T

    return df


def generate_comparison_table(df: pd.DataFrame, save_path: str):
    """生成对比表格（LaTeX格式和Markdown格式）"""

    # 提取均值列
    metric_cols = ['RMSE_mean', 'MAE_mean', 'PAI_mean',
                   'HR@10%_mean', 'Jaccard_mean']

    # 创建展示表格
    display_df = pd.DataFrame(index=df.index)
    for col in metric_cols:
        metric_name = col.replace('_mean', '')
        display_df[metric_name] = df.apply(
            lambda row: f"{row[col]:.4f}±{row[f'{metric_name}_std']:.4f}", axis=1
        )

    # 保存CSV
    display_df.to_csv(save_path.replace('.txt', '.csv'))

    # 生成Markdown表格
    with open(save_path.replace('.txt', '.md'), 'w') as f:
        f.write("# Baseline Comparison Results\n\n")
        f.write(display_df.to_markdown())
        f.write("\n\n")

        # 计算改进百分比
        f.write("## Improvement over Best Baseline\n\n")
        baseline_metrics = {}
        for model in ['HA', 'RF', 'ConvLSTM', 'STGCN', 'DCRNN', 'GraphWaveNet', 'STT_NoSemantic']:
            if model in df.index:
                baseline_metrics[model] = {
                    'RMSE': df.loc[model, 'RMSE_mean'],
                    'MAE': df.loc[model, 'MAE_mean'],
                    'PAI': df.loc[model, 'PAI_mean'],
                    'HR@10%': df.loc[model, 'HR@10%_mean'],
                    'Jaccard': df.loc[model, 'Jaccard_mean']
                }

        # 找最佳baseline
        best_baseline = min(baseline_metrics.keys(),
                           key=lambda x: baseline_metrics[x]['RMSE'])

        f.write(f"Best baseline: {best_baseline}\n\n")

        ours_metrics = {
            'RMSE': df.loc['Ours', 'RMSE_mean'],
            'MAE': df.loc['Ours', 'MAE_mean'],
            'PAI': df.loc['Ours', 'PAI_mean'],
            'HR@10%': df.loc['Ours', 'HR@10%_mean'],
            'Jaccard': df.loc['Ours', 'Jaccard_mean']
        }

        improvements = {}
        for metric in ['RMSE', 'MAE', 'PAI', 'HR@10%', 'Jaccard']:
            baseline_val = baseline_metrics[best_baseline][metric]
            ours_val = ours_metrics[metric]
            if metric in ['RMSE', 'MAE']:
                imp = (baseline_val - ours_val) / baseline_val * 100
            else:
                imp = (ours_val - baseline_val) / baseline_val * 100
            improvements[metric] = imp
            f.write(f"{metric}: {imp:+.2f}%\n")

    print(f"\nResults saved to:")
    print(f"  - CSV: {save_path.replace('.txt', '.csv')}")
    print(f"  - Markdown: {save_path.replace('.txt', '.md')}")

    return display_df


def main():
    """主函数"""
    print("=" * 80)
    print("Baseline Training and Evaluation Script")
    print("=" * 80)

    # 加载数据
    all_data = load_all_data()

    # 运行多次实验
    all_results = []
    for i, seed in enumerate(Config.SEEDS[:Config.NUM_RUNS]):
        print(f"\n\nRun {i+1}/{Config.NUM_RUNS}")
        results = run_single_experiment(seed, all_data)
        all_results.append(results)

        # 保存单次结果
        with open(f"{Config.RESULT_DIR}/run_{i+1}_seed{seed}.json", 'w') as f:
            json.dump(results, f, indent=2)

    # 聚合结果
    print("\n" + "=" * 80)
    print("Aggregating results...")
    df = aggregate_results(all_results)

    # 生成对比表格
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{Config.RESULT_DIR}/baseline_comparison_{timestamp}.txt"
    display_df = generate_comparison_table(df, save_path)

    # 打印结果
    print("\n" + "=" * 80)
    print("Final Results (Mean±Std):")
    print("=" * 80)
    print(display_df.to_string())

    # 保存完整结果
    df.to_csv(f"{Config.RESULT_DIR}/baseline_full_results_{timestamp}.csv")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print(f"All results saved to: {Config.RESULT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
