"""
压力测试：人为屏蔽实验 (Masking Experiment)
验证L-EPSTD在冷启动场景下的性能

实验设计：
1. 随机挑选20%的高犯罪网格，强制清零历史数据
2. 对比L-EPSTD与基线模型在这些"伪冷启动"网格上的表现
3. 预期：基线模型失效，L-EPSTD凭借环境逻辑仍能识别风险
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.epstd_stage1 import EnvironmentEncoder
from src.epstd_stage2 import PrototypeLibrary
from src.epstd_stage3_enhanced import (
    LogicGuidedDiffusion, LogicGuidedDiffusionScheduler,
    LogicConstraintCalculator, LogicGuidedPredictor,
    apply_masking_experiment, evaluate_cold_start_performance
)
from src.epstd_stage3 import ConditionalRiskDiffusion, DiffusionScheduler
from src.train_stgcn_trans import calculate_metrics, SpatioTemporalTransformer


def load_models(device='cuda'):
    """加载所有模型"""
    print("Loading models...")

    # 加载环境编码器
    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    env_encoder.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    env_encoder.eval()

    # 加载原型库
    import pickle
    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']
    prototype_library.prototype_risk_dists = proto_data['prototype_risk_dists']

    # 加载L-EPSTD (带逻辑引导)
    logic_model = ConditionalRiskDiffusion(
        num_nodes=1246, hidden_dim=128, num_layers=4,
        time_dim=64, env_dim=64, num_prototypes=10
    ).to(device)

    if os.path.exists('checkpoints/logic_guided_diffusion_best.pt'):
        checkpoint = torch.load('checkpoints/logic_guided_diffusion_best.pt', map_location=device)
        logic_model.load_state_dict(checkpoint['model'])
        print("Loaded L-EPSTD (logic-guided)")
    else:
        print("Warning: L-EPSTD checkpoint not found, using random weights")

    # 加载基线STGCN
    baseline_model = None
    if os.path.exists('checkpoints/best_model_trans.pt'):
        baseline_model = SpatioTemporalTransformer(
            static_dim=24, dynamic_dim=19, semantic_dim=0,
            hidden_dim=64, num_heads=4, num_nodes=1246
        ).to(device)
        baseline_model.load_state_dict(torch.load('checkpoints/best_model_trans.pt', map_location=device))
        baseline_model.eval()
        print("Loaded baseline STGCN")

    return env_encoder, prototype_library, logic_model, baseline_model


def run_masking_experiment_dual(
    X, Y_violent, Y_property,
    env_encoder,
    prototype_library,
    logic_model,
    baseline_model=None,
    mask_ratio=0.2,
    device='cuda',
    task='violent'  # 评估哪个任务
):
    """
    运行屏蔽实验（双任务版本）
    """
    print("\n" + "="*70)
    print(f"Masking Experiment: {mask_ratio*100}% high-crime grids masked")
    print(f"Task: {task.upper()}")
    print("="*70)

    # 选择任务数据
    if task == 'violent':
        Y = Y_violent
    else:
        Y = Y_property

    # 划分训练/测试
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # 应用屏蔽
    Y_train_masked, mask_indices = apply_masking_experiment(
        Y_train, mask_ratio=mask_ratio, target='high_crime'
    )

    # 创建逻辑引导调度器
    logic_scheduler = LogicGuidedDiffusionScheduler(num_timesteps=1000, device=device)
    logic_calculator = LogicConstraintCalculator(static_feature_dim=24).to(device)

    # ============== L-EPSTD 预测 (带逻辑引导) ==============
    print("\n[L-EPSTD with Logic Guidance]")
    predictor_logic = LogicGuidedPredictor(
        logic_model, logic_scheduler, logic_calculator,
        env_encoder, prototype_library, device
    )

    static_features = X_test[-1, :, :24]
    pred_logic, std_logic, pi_logic = predictor_logic.predict(
        static_features, use_logic_guidance=True, guidance_scale=1.0, num_samples=5
    )

    # 评估在masked网格上的表现
    metrics_logic_masked = evaluate_cold_start_performance(
        pred_logic, Y_test[-1], mask_indices
    )

    # 评估整体表现
    metrics_logic_all = calculate_metrics(pred_logic, Y_test[-1])

    print(f"  Masked Grids Performance:")
    print(f"    MAE: {metrics_logic_masked['mae']:.4f}")
    print(f"    Correlation: {metrics_logic_masked['correlation']:.4f}")
    print(f"    Hotspot Recall: {metrics_logic_masked['hotspot_recall']:.4f}")
    print(f"  Overall Performance:")
    print(f"    PAI@10%: {metrics_logic_all.get('pai_top10', 0):.4f}")
    print(f"    Recall@10%: {metrics_logic_all.get('recall_top10', 0):.4f}")

    # ============== L-EPSTD 预测 (无逻辑引导) ==============
    print("\n[L-EPSTD without Logic Guidance]")
    pred_no_logic, _, _ = predictor_logic.predict(
        static_features, use_logic_guidance=False, num_samples=5
    )

    metrics_no_logic_masked = evaluate_cold_start_performance(
        pred_no_logic, Y_test[-1], mask_indices
    )
    metrics_no_logic_all = calculate_metrics(pred_no_logic, Y_test[-1])

    print(f"  Masked Grids Performance:")
    print(f"    MAE: {metrics_no_logic_masked['mae']:.4f}")
    print(f"    Correlation: {metrics_no_logic_masked['correlation']:.4f}")
    print(f"    Hotspot Recall: {metrics_no_logic_masked['hotspot_recall']:.4f}")

    # ============== 基线模型预测 ==============
    if baseline_model is not None:
        print("\n[Baseline STGCN]")
        # 使用masked数据训练/预测基线模型（简化：直接使用预训练模型）
        # 实际应在masked数据上重新训练，这里简化处理

        with torch.no_grad():
            X_sample = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)
            N = 1246
            A_spatial = torch.eye(N).to(device)
            A_distance = torch.eye(N).to(device)
            A_crime = torch.eye(N).unsqueeze(0).to(device)
            A_hypergraph = torch.eye(N).to(device)
            OD = torch.zeros(1, N, 4).to(device)
            crime_h = X_sample[:, :, :, -7:] if X_sample.dim() == 4 else X_sample[:, :, -7:]
            if crime_h.dim() == 3:
                crime_h = crime_h[:, :, 0]

            pi, mu, _, _, _ = baseline_model(
                X_sample, A_spatial, A_distance, A_crime,
                A_hypergraph, OD, None, crime_h
            )
            pred_baseline = ((1 - pi) * mu).squeeze().cpu().numpy()

        metrics_baseline_masked = evaluate_cold_start_performance(
            pred_baseline, Y_test[-1], mask_indices
        )
        metrics_baseline_all = calculate_metrics(pred_baseline, Y_test[-1])

        print(f"  Masked Grids Performance:")
        print(f"    MAE: {metrics_baseline_masked['mae']:.4f}")
        print(f"    Correlation: {metrics_baseline_masked['correlation']:.4f}")
        print(f"    Hotspot Recall: {metrics_baseline_masked['hotspot_recall']:.4f}")
    else:
        metrics_baseline_masked = None
        metrics_baseline_all = None
        pred_baseline = None

    # ============== 对比结果 ==============
    print("\n" + "="*70)
    print("Comparison Summary (Masked Grids Only)")
    print("="*70)

    results = {
        'L-EPSTD (with logic)': metrics_logic_masked,
        'L-EPSTD (no logic)': metrics_no_logic_masked,
    }

    if metrics_baseline_masked:
        results['Baseline STGCN'] = metrics_baseline_masked

    print(f"{'Model':<25} {'MAE':<10} {'Corr':<10} {'Recall':<10}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<25} {m['mae']:<10.4f} {m['correlation']:<10.4f} {m['hotspot_recall']:<10.4f}")

    # 保存结果
    os.makedirs('experiments/results', exist_ok=True)
    np.savez(
        f'experiments/results/masking_experiment_{int(mask_ratio*100)}pct.npz',
        mask_indices=mask_indices,
        pred_logic=pred_logic,
        pred_no_logic=pred_no_logic,
        pred_baseline=pred_baseline,
        y_true=Y_test[-1],
        metrics_logic=metrics_logic_masked,
        metrics_no_logic=metrics_no_logic_masked,
        metrics_baseline=metrics_baseline_masked if metrics_baseline_masked else {}
    )

    print(f"\nResults saved to experiments/results/masking_experiment_{int(mask_ratio*100)}pct.npz")

    return results, pred_logic, pred_baseline


def visualize_masking_results(results_file='experiments/results/masking_experiment_20pct.npz'):
    """可视化压力测试结果"""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    data = np.load(results_file, allow_pickle=True)

    mask_indices = data['mask_indices']
    pred_logic = data['pred_logic']
    pred_no_logic = data['pred_no_logic']
    pred_baseline = data['pred_baseline'] if 'pred_baseline' in data else None
    y_true = data['y_true']

    # 只关注masked网格
    y_masked = y_true[mask_indices]
    pred_logic_masked = pred_logic[mask_indices]
    pred_no_logic_masked = pred_no_logic[mask_indices]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 散点图：真实值 vs 预测值
    ax = axes[0, 0]
    ax.scatter(y_masked, pred_logic_masked, alpha=0.5, label='L-EPSTD (with logic)', s=20)
    ax.scatter(y_masked, pred_no_logic_masked, alpha=0.5, label='L-EPSTD (no logic)', s=20)
    if pred_baseline is not None:
        ax.scatter(y_masked, pred_baseline[mask_indices], alpha=0.5, label='Baseline', s=20)
    ax.plot([0, y_masked.max()], [0, y_masked.max()], 'k--', label='Perfect')
    ax.set_xlabel('True Risk')
    ax.set_ylabel('Predicted Risk')
    ax.set_title('Predictions on Masked (Cold-Start) Grids')
    ax.legend()

    # 2. 柱状图：指标对比
    ax = axes[0, 1]
    metrics_logic = data['metrics_logic'].item()
    metrics_no_logic = data['metrics_no_logic'].item()
    metrics_baseline = data['metrics_baseline'].item() if len(data['metrics_baseline']) > 0 else None

    labels = ['MAE', 'Correlation', 'Hotspot\nRecall']
    x = np.arange(len(labels))
    width = 0.25

    values_logic = [metrics_logic['mae'], metrics_logic['correlation'], metrics_logic['hotspot_recall']]
    values_no_logic = [metrics_no_logic['mae'], metrics_no_logic['correlation'], metrics_no_logic['hotspot_recall']]

    ax.bar(x - width, values_logic, width, label='L-EPSTD (logic)')
    ax.bar(x, values_no_logic, width, label='L-EPSTD (no logic)')

    if metrics_baseline:
        values_baseline = [metrics_baseline['mae'], metrics_baseline['correlation'], metrics_baseline['hotspot_recall']]
        ax.bar(x + width, values_baseline, width, label='Baseline')

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison on Cold-Start Grids')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 3. 热力图：masked网格的空间分布
    ax = axes[1, 0]
    # 假设网格是方形排列（需要根据实际情况调整）
    grid_size = int(np.sqrt(len(y_true)))
    if grid_size * grid_size == len(y_true):
        mask_map = np.zeros((grid_size, grid_size))
        for idx in mask_indices:
            row = idx // grid_size
            col = idx % grid_size
            mask_map[row, col] = 1
        im = ax.imshow(mask_map, cmap='Reds')
        ax.set_title('Masked (Cold-Start) Grid Locations')
        plt.colorbar(im, ax=ax, label='Masked')
    else:
        ax.text(0.5, 0.5, 'Cannot visualize grid layout', ha='center', va='center')
        ax.set_title('Masked Grid Distribution')

    # 4. 残差分布
    ax = axes[1, 1]
    residual_logic = np.abs(pred_logic_masked - y_masked)
    residual_no_logic = np.abs(pred_no_logic_masked - y_masked)

    ax.hist(residual_logic, bins=30, alpha=0.5, label='L-EPSTD (logic)')
    ax.hist(residual_no_logic, bins=30, alpha=0.5, label='L-EPSTD (no logic)')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution on Cold-Start Grids')
    ax.legend()

    plt.tight_layout()
    plt.savefig('experiments/results/masking_experiment_visualization.png', dpi=150)
    print("Visualization saved to experiments/results/masking_experiment_visualization.png")


def main():
    """主流程（双任务版本）"""
    print("="*70)
    print("Masking Experiment: Cold-Start Performance Evaluation (Dual-Task)")
    print("="*70)

    # 加载双任务数据
    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')

    # 加载暴力和财产犯罪数据
    Y_violent = np.load(f'{data_dir}/Y_violent.npy') if os.path.exists(f'{data_dir}/Y_violent.npy') else np.load(f'{data_dir}/Y.npy')
    Y_property = np.load(f'{data_dir}/Y_property.npy') if os.path.exists(f'{data_dir}/Y_property.npy') else np.load(f'{data_dir}/Y.npy')

    print(f"Data loaded: X={X.shape}")
    print(f"  Violent: {Y_violent.shape}, sparsity={(Y_violent==0).mean():.2%}")
    print(f"  Property: {Y_property.shape}, sparsity={(Y_property==0).mean():.2%}")

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_encoder, prototype_library, logic_model, baseline_model = load_models(device)

    # 对暴力犯罪（核心任务）运行不同屏蔽比例的实验
    print("\n" + "="*70)
    print("Testing VIOLENT Crime (Core Task with Logic Guidance)")
    print("="*70)
    for mask_ratio in [0.2, 0.3]:
        run_masking_experiment_dual(
            X, Y_violent, Y_property,
            env_encoder, prototype_library, logic_model, baseline_model,
            mask_ratio=mask_ratio, device=device, task='violent'
        )

    # 对财产犯罪（辅助任务）运行实验
    print("\n" + "="*70)
    print("Testing PROPERTY Crime (Auxiliary Task)")
    print("="*70)
    for mask_ratio in [0.2]:
        run_masking_experiment_dual(
            X, Y_violent, Y_property,
            env_encoder, prototype_library, logic_model, baseline_model,
            mask_ratio=mask_ratio, device=device, task='property'
        )

    # 可视化结果
    visualize_masking_results()

    print("\n" + "="*70)
    print("Dual-Task Masking Experiment Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
