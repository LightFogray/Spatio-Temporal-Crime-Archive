"""
EP-STD Ablation Study
======================
EP-STD模型的完整消融实验

消融变体:
  1. Full_L-EPSTD: 完整模型（含逻辑引导）
  2. w_o_Logic_Guidance: 移除T-Norm逻辑引导
  3. w_o_Adaptive_Fusion: 移除环境自适应图融合（改为固定权重）
  4. w_o_CrossCrime_Gate: 移除双犯罪交叉门控
  5. w_o_MultiGraph: 仅使用单图（空间邻接图）
  6. w_o_Environment_Encoder: 移除环境编码器（随机初始化）
  7. w_o_ZeroInflation: 移除零膨胀建模
  8. w_o_Prototype: 移除原型学习引导

评估指标:
  - 全局性能: MAE, Correlation, PAI, Recall@10%
  - 冷启动性能: Cold-Start MAE, Cold-Start Recall
  - 计算效率: Inference Time, GPU Memory
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.epstd_stage1 import EnvironmentEncoder
from src.epstd_stage2 import PrototypeLibrary
from src.epstd_stage3_multigraph import (
    MultiGraphConditionalDiffusion,
    DualTaskMultiGraphDiffusion,
    AdaptiveGraphFusion,
    CrossCrimeGate,
    MultiGraphAttentionLayer
)
from src.epstd_stage3_enhanced import (
    LogicGuidedDiffusionScheduler,
    LogicConstraintCalculator
)


# ================================
# 配置参数
# ================================
class AblationConfig:
    """消融实验配置"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
    CHECKPOINT_DIR = "checkpoints/ablation"
    RESULT_DIR = "experiments/results"

    EPOCHS = 50  # 消融实验减少epoch数
    BATCH_SIZE = 16
    LR = 1e-4
    PATIENCE = 10

    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    DROPOUT = 0.1

    SEEDS = [42, 123, 456]  # 多次运行取平均
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 评估设置
    MASK_RATIOS = [0.2, 0.3]  # 冷启动测试的屏蔽比例


os.makedirs(AblationConfig.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AblationConfig.RESULT_DIR, exist_ok=True)


# ================================
# 消融变体定义
# ================================

ABLATION_VARIANTS = {
    # 基准完整模型
    'Full_L-EPSTD': {
        'description': '完整L-EPSTD模型（含逻辑引导+自适应融合+交叉门控）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 1. 逻辑引导消融
    'w_o_Logic_Guidance': {
        'description': '移除T-Norm逻辑引导（仅保留扩散损失）',
        'use_logic_guidance': False,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.0}
    },

    # 2. 自适应图融合消融
    'w_o_Adaptive_Fusion': {
        'description': '移除环境自适应图融合（改为固定等权重融合）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': False,  # 使用固定权重
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 3. 交叉犯罪门控消融
    'w_o_CrossCrime_Gate': {
        'description': '移除双犯罪交叉门控（仅用Violent犯罪图）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': False,  # 禁用门控，只用Violent图
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 4. 多图结构消融
    'w_o_MultiGraph': {
        'description': '仅使用单图（仅空间邻接图，移除其他4张图）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': False,  # 只用1张图
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 5. 环境编码器消融
    'w_o_Environment_Encoder': {
        'description': '移除环境编码器（使用随机初始化的静态特征）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': False,  # 不使用预训练编码器
        'use_zero_inflation': True,
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 6. 零膨胀建模消融
    'w_o_ZeroInflation': {
        'description': '移除零膨胀建模（直接使用MSE损失）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': False,  # 禁用ZI
        'use_prototype': True,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.0, 'logic': 0.5}
    },

    # 7. 原型学习消融
    'w_o_Prototype': {
        'description': '移除原型学习引导（禁用原型嵌入）',
        'use_logic_guidance': True,
        'use_adaptive_fusion': True,
        'use_cross_crime_gate': True,
        'use_multi_graph': True,
        'use_env_encoder': True,
        'use_zero_inflation': True,
        'use_prototype': False,  # 禁用原型
        'loss_weights': {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5}
    },

    # 8. 极简单模型（仅保留基础扩散）
    'Base_Diffusion_Only': {
        'description': '最简基线（仅基础扩散，移除所有创新组件）',
        'use_logic_guidance': False,
        'use_adaptive_fusion': False,
        'use_cross_crime_gate': False,
        'use_multi_graph': False,
        'use_env_encoder': False,
        'use_zero_inflation': False,
        'use_prototype': False,
        'loss_weights': {'diffusion': 1.0, 'zi': 0.0, 'logic': 0.0}
    }
}


# ================================
# 消融模型包装器
# ================================

class AblationModelWrapper(nn.Module):
    """
    消融实验专用模型包装器
    根据配置动态调整模型结构和行为
    """

    def __init__(self, base_model, config: dict, env_encoder=None):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.env_encoder = env_encoder

        # 如果不是用环境编码器，创建一个简单的投影层
        if not config.get('use_env_encoder', True) or env_encoder is None:
            self.env_proj = nn.Sequential(
                nn.Linear(24, 64),
                nn.GELU(),
                nn.Linear(64, 64)
            )

        # 如果不是自适应融合，创建固定权重
        if not config.get('use_adaptive_fusion', True):
            self.register_buffer('fixed_graph_weights',
                                torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))

    def forward(self, x_t, t, env_emb, prototype_ids, adj_list=None, crime_stats=None):
        """
        前向传播，根据配置启用/禁用特定功能
        """
        # 如果不是用环境编码器，使用简单投影
        if not self.config.get('use_env_encoder', True) and self.env_encoder is None:
            # env_emb 应该是原始静态特征
            env_emb = self.env_proj(env_emb)

        # 如果不是用多图，只保留第一张图
        if not self.config.get('use_multi_graph', True) and adj_list is not None:
            adj_list = [adj_list[0]]  # 只用空间邻接图

        # 如果不是交叉犯罪门控，修改crime_stats使门控偏向violent
        if not self.config.get('use_cross_crime_gate', True) and crime_stats is not None:
            # 强制门控为 [1.0, 0.0]，只用violent图
            B, N, _ = crime_stats.shape
            crime_stats = torch.zeros(B, N, 2, device=crime_stats.device)
            crime_stats[:, :, 0] = 1.0  # violent权重为1

        # 调用基础模型
        outputs = self.base_model(x_t, t, env_emb, prototype_ids, adj_list, crime_stats)

        # 如果不是零膨胀建模，修改输出
        if not self.config.get('use_zero_inflation', True):
            noise_pred, pi, graph_weights, crime_gates = outputs
            pi = torch.zeros_like(pi)  # ZI概率为0
            outputs = (noise_pred, pi, graph_weights, crime_gates)

        return outputs


# ================================
# 评估指标计算
# ================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_percent: float = 0.1) -> Dict:
    """计算评估指标"""
    from scipy import stats

    # 展平
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 基础指标
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))

    # 相关性
    if len(np.unique(y_true_flat)) > 1 and len(np.unique(y_pred_flat)) > 1:
        correlation, _ = stats.pearsonr(y_true_flat, y_pred_flat)
    else:
        correlation = 0.0

    # PAI (Prediction Accuracy Index)
    k = int(len(y_true_flat) * k_percent)
    top_k_pred = np.argsort(y_pred_flat)[-k:]
    top_k_true = np.argsort(y_true_flat)[-k:]

    hits = len(set(top_k_pred) & set(top_k_true))
    hit_rate = hits / k

    # PAI = Hit Rate / (Area Ratio)
    pai = hit_rate / k_percent

    # Recall@k%
    recall = hits / len(top_k_true) if len(top_k_true) > 0 else 0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'PAI': pai,
        f'Recall@{int(k_percent*100)}%': recall,
        'HitRate': hit_rate
    }


def evaluate_cold_start(y_true: np.ndarray, y_pred: np.ndarray,
                       mask_indices: np.ndarray) -> Dict:
    """评估冷启动性能（仅在屏蔽网格上计算）"""
    y_true_masked = y_true[mask_indices]
    y_pred_masked = y_pred[mask_indices]

    metrics = calculate_metrics(y_true_masked, y_pred_masked)

    # 重命名为冷启动指标
    return {
        'CS_MAE': metrics['MAE'],
        'CS_RMSE': metrics['RMSE'],
        'CS_Correlation': metrics['Correlation'],
        'CS_PAI': metrics['PAI'],
        'CS_Recall': metrics['Recall@10%']
    }


# ================================
# 训练函数
# ================================

def train_ablation_variant(
    variant_name: str,
    variant_config: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    adj_list: List[np.ndarray],
    env_encoder: EnvironmentEncoder,
    prototype_library: PrototypeLibrary,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    训练单个消融变体

    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    print(f"\n{'='*70}")
    print(f"Training: {variant_name}")
    print(f"Description: {variant_config['description']}")
    print(f"{'='*70}")

    num_nodes = Y_train.shape[1]
    num_prototypes = prototype_library.n_prototypes

    # 创建基础模型
    base_model = MultiGraphConditionalDiffusion(
        num_nodes=num_nodes,
        hidden_dim=AblationConfig.HIDDEN_DIM,
        num_layers=AblationConfig.NUM_LAYERS,
        time_dim=64,
        env_dim=64,
        num_prototypes=num_prototypes if variant_config.get('use_prototype', True) else 1
    ).to(device)

    # 包装为消融模型
    model = AblationModelWrapper(base_model, variant_config, env_encoder).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AblationConfig.LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=AblationConfig.EPOCHS
    )

    # 扩散调度器
    diffusion_scheduler = LogicGuidedDiffusionScheduler(
        num_timesteps=1000, device=device
    )

    # 逻辑计算器（如果需要）
    logic_calculator = None
    if variant_config.get('use_logic_guidance', True):
        logic_calculator = LogicConstraintCalculator(
            static_feature_dim=24
        ).to(device)

    # 准备数据
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # 邻接矩阵
    adj_tensors = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj_list]

    # 原型标签
    prototype_labels = np.load(f'{AblationConfig.DATA_DIR}/prototype_labels.npy')
    prototype_ids = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(AblationConfig.EPOCHS):
        model.train()
        epoch_losses = []

        # 逻辑引导强度调度
        if epoch < 20:
            guidance_scale = 0.0
        else:
            guidance_scale = min(1.0, (epoch - 20) / 20.0) if variant_config.get('use_logic_guidance', True) else 0.0

        # 批量训练
        num_samples = len(X_train)
        indices = np.random.permutation(num_samples)

        for i in range(0, num_samples, AblationConfig.BATCH_SIZE):
            batch_idx = indices[i:i + AblationConfig.BATCH_SIZE]
            if len(batch_idx) < 2:
                continue

            x_0 = Y_train_tensor[batch_idx]

            # 获取环境嵌入
            if variant_config.get('use_env_encoder', True):
                with torch.no_grad():
                    env_emb = env_encoder(X_train_tensor[batch_idx])
            else:
                # 使用原始特征
                env_emb = X_train_tensor[batch_idx]

            proto_ids_batch = prototype_ids.unsqueeze(0).expand(len(batch_idx), -1)

            # 前向扩散
            t = diffusion_scheduler.sample_timesteps(len(batch_idx))
            noise = torch.randn_like(x_0)
            x_t = diffusion_scheduler.add_noise(x_0, t, noise)

            # 计算crime_stats
            crime_stats = torch.zeros(len(batch_idx), num_nodes, 2, device=device)
            crime_stats[:, :, 0] = x_0.mean(dim=1, keepdim=True).expand(-1, num_nodes)

            # 预测
            noise_pred, pi, graph_weights, crime_gates = model(
                x_t, t, env_emb, proto_ids_batch, adj_tensors, crime_stats
            )

            # 计算损失
            loss_diffusion = F.mse_loss(noise_pred, noise)

            # 零膨胀损失
            if variant_config.get('use_zero_inflation', True):
                zero_mask = (x_0 == 0).float()
                loss_zi = F.binary_cross_entropy(pi, zero_mask)
            else:
                loss_zi = torch.tensor(0.0, device=device)

            # 逻辑引导损失
            loss_logic = torch.tensor(0.0, device=device)
            if guidance_scale > 0 and logic_calculator is not None:
                # 简化的逻辑损失计算
                static_features = X_train_tensor[batch_idx, :, :24]
                logic_loss = logic_calculator(x_t, static_features, noise_pred)
                loss_logic = logic_loss * guidance_scale

            # 总损失
            weights = variant_config.get('loss_weights', {'diffusion': 1.0, 'zi': 0.1, 'logic': 0.5})
            loss = (weights['diffusion'] * loss_diffusion +
                   weights['zi'] * loss_zi +
                   weights['logic'] * loss_logic)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []

        with torch.no_grad():
            for i in range(0, len(X_val), AblationConfig.BATCH_SIZE):
                batch_idx = list(range(i, min(i + AblationConfig.BATCH_SIZE, len(X_val))))
                if len(batch_idx) < 2:
                    continue

                x_0 = Y_val_tensor[batch_idx]

                if variant_config.get('use_env_encoder', True):
                    env_emb = env_encoder(X_val_tensor[batch_idx])
                else:
                    env_emb = X_val_tensor[batch_idx]

                proto_ids_batch = prototype_ids.unsqueeze(0).expand(len(batch_idx), -1)

                t = diffusion_scheduler.sample_timesteps(len(batch_idx))
                noise = torch.randn_like(x_0)
                x_t = diffusion_scheduler.add_noise(x_0, t, noise)

                crime_stats = torch.zeros(len(batch_idx), num_nodes, 2, device=device)
                crime_stats[:, :, 0] = x_0.mean(dim=1, keepdim=True).expand(-1, num_nodes)

                noise_pred, pi, _, _ = model(x_t, t, env_emb, proto_ids_batch, adj_tensors, crime_stats)

                loss = F.mse_loss(noise_pred, noise)
                val_losses.append(loss.item())

        mean_train = np.mean(epoch_losses) if epoch_losses else float('inf')
        mean_val = np.mean(val_losses) if val_losses else float('inf')

        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{AblationConfig.EPOCHS} | "
                  f"Train: {mean_train:.4f} | Val: {mean_val:.4f} | "
                  f"Guidance: {guidance_scale:.2f}")

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

    # 加载最佳模型
    model.load_state_dict(torch.load(f"{AblationConfig.CHECKPOINT_DIR}/{variant_name}_best.pt"))

    return model, history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    adj_list: List[np.ndarray],
    env_encoder: EnvironmentEncoder,
    prototype_library: PrototypeLibrary,
    variant_config: dict,
    device: str = 'cuda'
) -> Dict:
    """评估模型性能"""

    num_nodes = Y_test.shape[1]

    # 准备数据
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
    adj_tensors = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in adj_list]

    prototype_labels = np.load(f'{AblationConfig.DATA_DIR}/prototype_labels.npy')
    prototype_ids = torch.tensor(prototype_labels, dtype=torch.long).to(device)

    diffusion_scheduler = LogicGuidedDiffusionScheduler(num_timesteps=1000, device=device)

    model.eval()
    all_preds = []

    # 推理时间测试
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(X_test), AblationConfig.BATCH_SIZE):
            batch_idx = list(range(i, min(i + AblationConfig.BATCH_SIZE, len(X_test))))
            if len(batch_idx) < 1:
                continue

            # 获取环境嵌入
            if variant_config.get('use_env_encoder', True):
                env_emb = env_encoder(X_test_tensor[batch_idx])
            else:
                env_emb = X_test_tensor[batch_idx]

            proto_ids_batch = prototype_ids.unsqueeze(0).expand(len(batch_idx), -1)

            # 从噪声开始采样
            x_t = torch.randn(len(batch_idx), num_nodes, device=device)

            # DDPM采样（简化版）
            for t in range(999, -1, -50):  # 每50步采样一次
                t_batch = torch.tensor([t] * len(batch_idx), device=device)

                crime_stats = torch.zeros(len(batch_idx), num_nodes, 2, device=device)

                noise_pred, pi, _, _ = model(x_t, t_batch, env_emb, proto_ids_batch, adj_tensors, crime_stats)

                # 简化的去噪步骤
                alpha_t = diffusion_scheduler.alphas[t]
                alpha_bar_t = diffusion_scheduler.alphas_cumprod[t]

                x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = x_t + torch.sqrt(diffusion_scheduler.betas[t]) * noise

            # 最终预测
            pred = torch.clamp(x_t, min=0)
            all_preds.append(pred.cpu().numpy())

    inference_time = time.time() - start_time

    y_pred = np.vstack(all_preds)
    y_true = Y_test

    # 计算全局指标
    metrics = calculate_metrics(y_true, y_pred)

    # 计算冷启动指标（模拟）
    np.random.seed(42)
    mask_indices = np.random.choice(num_nodes, size=int(num_nodes * 0.2), replace=False)
    cs_metrics = evaluate_cold_start(y_true[-1], y_pred[-1], mask_indices)

    # 合并指标
    metrics.update(cs_metrics)
    metrics['Inference_Time'] = inference_time / len(X_test)  # 每个样本的平均时间

    return metrics


# ================================
# 报告生成
# ================================

def generate_ablation_report(all_results: Dict, save_path: str):
    """生成消融实验报告"""

    # 创建DataFrame
    df = pd.DataFrame(all_results).T

    # 计算相对于Full Model的性能变化
    if 'Full_L-EPSTD' in df.index:
        full_metrics = df.loc['Full_L-EPSTD']

        # 计算性能下降百分比
        for col in ['PAI', 'Correlation', 'CS_PAI', 'CS_Recall']:
            if col in df.columns:
                df[f'{col}_drop'] = df.apply(
                    lambda row: ((full_metrics[col] - row[col]) / full_metrics[col] * 100)
                    if row.name != 'Full_L-EPSTD' else 0, axis=1
                )

        for col in ['MAE', 'RMSE', 'CS_MAE']:
            if col in df.columns:
                df[f'{col}_increase'] = df.apply(
                    lambda row: ((row[col] - full_metrics[col]) / full_metrics[col] * 100)
                    if row.name != 'Full_L-EPSTD' else 0, axis=1
                )

    # 保存CSV
    csv_path = save_path.replace('.txt', '.csv')
    df.to_csv(csv_path)

    # 生成Markdown报告
    md_path = save_path.replace('.txt', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# EP-STD Ablation Study Results\n\n")
        f.write("## Experiment Configuration\n\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- **Device**: {AblationConfig.DEVICE}\n")
        f.write(f"- **Epochs**: {AblationConfig.EPOCHS}\n")
        f.write(f"- **Batch Size**: {AblationConfig.BATCH_SIZE}\n\n")

        f.write("## Results Summary\n\n")

        # 主要指标表
        main_cols = ['MAE', 'Correlation', 'PAI', 'CS_MAE', 'CS_Recall', 'Inference_Time']
        main_df = df[[col for col in main_cols if col in df.columns]]
        f.write(main_df.to_markdown())
        f.write("\n\n")

        # 组件贡献分析
        f.write("## Component Contribution Analysis\n\n")
        f.write("Relative to Full Model:\n\n")

        for variant in ABLATION_VARIANTS.keys():
            if variant != 'Full_L-EPSTD' and variant in df.index:
                pai_drop = df.loc[variant, 'PAI_drop'] if 'PAI_drop' in df.columns else 0
                cs_recall_drop = df.loc[variant, 'CS_Recall_drop'] if 'CS_Recall_drop' in df.columns else 0
                f.write(f"- **{variant}**:\n")
                f.write(f"  - PAI drop: {pai_drop:.2f}%\n")
                f.write(f"  - Cold-start Recall drop: {cs_recall_drop:.2f}%\n")

        # 关键发现
        f.write("\n## Key Findings\n\n")

        # 找出最关键的组件
        if 'CS_Recall_drop' in df.columns:
            sorted_variants = df[df.index != 'Full_L-EPSTD']['CS_Recall_drop'].sort_values(ascending=False)
            f.write("### Most Critical Components (by Cold-Start Impact)\n\n")
            for variant, drop in sorted_variants.head(3).items():
                f.write(f"1. **{variant}**: {drop:.2f}% drop in CS Recall\n")

    print(f"\nAblation report saved to:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Markdown: {md_path}")

    return df


# ================================
# 主流程
# ================================

def load_data():
    """加载数据"""
    print("Loading data...")

    data_dir = AblationConfig.DATA_DIR

    X = np.load(f"{data_dir}/X.npy")
    Y = np.load(f"{data_dir}/Y.npy")

    # 处理Y的形状
    if Y.ndim == 3:
        Y = Y[:, :, 0]  # 使用暴力犯罪

    # 提取静态特征
    X_static = X[:, :, :24]

    # 加载图
    adj_list = [
        np.load(f"{data_dir}/adj_adaptive.npy"),
        np.load(f"{data_dir}/adj_distance.npy"),
        np.load(f"{data_dir}/adj_crime_violent.npy"),
        np.load(f"{data_dir}/adj_crime_property.npy"),
        np.load(f"{data_dir}/adj_od.npy")
    ]

    # 处理动态图
    for i in [2, 3]:
        if len(adj_list[i].shape) == 3:
            adj_list[i] = adj_list[i].mean(axis=0)

    print(f"X shape: {X_static.shape}")
    print(f"Y shape: {Y.shape}")

    return X_static, Y, adj_list


def main():
    """主函数"""
    print("=" * 80)
    print("EP-STD Ablation Study")
    print("=" * 80)
    print(f"Device: {AblationConfig.DEVICE}")
    print(f"Variants: {len(ABLATION_VARIANTS)}")
    print("=" * 80)

    # 加载数据
    X, Y, adj_list = load_data()

    # 划分数据
    n_samples = len(X)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    Y_train, Y_val, Y_test = Y[:train_end], Y[train_end:val_end], Y[val_end:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 加载预训练模型
    device = AblationConfig.DEVICE

    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    env_encoder.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    env_encoder.eval()

    import pickle
    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']

    # 运行所有消融实验
    all_results = {}

    for variant_name, variant_config in ABLATION_VARIANTS.items():
        # 训练
        model, history = train_ablation_variant(
            variant_name, variant_config,
            X_train, Y_train, X_val, Y_val,
            adj_list, env_encoder, prototype_library,
            device
        )

        # 评估
        metrics = evaluate_model(
            model, X_test, Y_test, adj_list,
            env_encoder, prototype_library,
            variant_config, device
        )

        all_results[variant_name] = metrics

        print(f"\nResults for {variant_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{AblationConfig.RESULT_DIR}/epstd_ablation_{timestamp}.txt"

    df = generate_ablation_report(all_results, save_path)

    # 保存JSON
    json_path = f"{AblationConfig.RESULT_DIR}/epstd_ablation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总
    print("\n" + "=" * 80)
    print("Ablation Study Summary")
    print("=" * 80)
    print(df[['MAE', 'Correlation', 'PAI', 'CS_MAE', 'CS_Recall']].to_string())

    print("\n" + "=" * 80)
    print("Ablation study completed!")
    print(f"Results saved to: {AblationConfig.RESULT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
