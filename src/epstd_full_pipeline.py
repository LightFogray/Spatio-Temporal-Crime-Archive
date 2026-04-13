"""
EP-STD 完整训练流程：Stage 1 + Stage 2 + Stage 3

架构概述：
=========
EP-STD (Environment-Prompted Spatio-Temporal Diffusion)
通过三阶段架构解决犯罪预测的冷启动和零膨胀问题：

Stage 1: 对比学习环境编码器
  - 学习目标：环境相似 -> 嵌入相近
  - 解决：环境表征学习

Stage 2: 原型学习与风险先验
  - 学习目标：环境聚类 -> 风险原型
  - 解决：冷启动预测、风险先验分布

Stage 3: 条件扩散模型
  - 学习目标：环境条件 -> 风险分布生成
  - 解决：不确定性估计、零膨胀建模、时空一致性

数据流：
========
原始特征 (N, 24)
    ↓
[Stage 1: EnvironmentEncoder]
环境嵌入 (N, 64)
    ↓
[Stage 2: PrototypeLibrary]
原型标签 (N,) + 风险先验 (K,)
    ↓
[Stage 3: ConditionalRiskDiffusion]
风险分布预测 (N,) + 不确定性 (N,) + 零膨胀概率 (N,)

使用示例：
=========
# 完整训练
python src/epstd_full_pipeline.py --stage all --epochs 100

# 单独运行Stage 3（假设1和2已完成）
python src/epstd_full_pipeline.py --stage 3 --epochs 100

# 仅评估
python src/epstd_full_pipeline.py --eval-only
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm


def run_stage1(epochs=100, device='cuda'):
    """运行Stage 1：对比学习环境编码器"""
    print("\n" + "="*70)
    print("EP-STD Stage 1: Contrastive Environment Encoder")
    print("="*70)

    from epstd_stage1 import (
        EnvironmentEncoder, train_contrastive_encoder,
        evaluate_encoder, visualize_embeddings
    )

    # 加载数据
    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    static_features = X[-1, :, :24]
    risk_labels = Y[-1, :]

    print(f"Loaded {len(static_features)} grids with {static_features.shape[1]} features")

    # 训练编码器
    model = train_contrastive_encoder(
        static_features, risk_labels,
        output_dim=64,
        epochs=epochs,
        lr=1e-3,
        batch_size=64,
        device=device
    )

    # 评估
    model.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    embeddings, metrics = evaluate_encoder(model, static_features, risk_labels, device)

    # 保存
    np.save(f'{data_dir}/env_embeddings.npy', embeddings)
    print(f"\nStage 1 completed! Embeddings shape: {embeddings.shape}")

    return model, embeddings


def run_stage2(device='cuda'):
    """运行Stage 2：原型学习"""
    print("\n" + "="*70)
    print("EP-STD Stage 2: Prototype Learning")
    print("="*70)

    from epstd_stage1 import EnvironmentEncoder
    from epstd_stage2 import PrototypeLibrary, visualize_prototypes, test_cold_start_prediction

    # 加载数据
    data_dir = 'data/processed'
    env_embeddings = np.load(f'{data_dir}/env_embeddings.npy')
    static_features = np.load(f'{data_dir}/X.npy')[-1, :, :24]
    risk_labels = np.load(f'{data_dir}/Y.npy')[-1, :]

    # 加载编码器
    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    checkpoint_path = 'checkpoints/env_encoder_best.pt'
    if os.path.exists(checkpoint_path):
        env_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    env_encoder.eval()

    # 构建原型库
    n_prototypes = 10
    prototype_lib = PrototypeLibrary(n_prototypes=n_prototypes)
    proto_labels = prototype_lib.fit(env_embeddings, risk_labels)

    # 可视化
    visualize_prototypes(env_embeddings, risk_labels, proto_labels)

    # 测试冷启动
    test_results = test_cold_start_prediction(
        prototype_lib, env_embeddings, risk_labels,
        static_features, env_encoder, test_ratio=0.2
    )

    # 保存
    import pickle
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/prototype_library.pkl', 'wb') as f:
        pickle.dump({
            'prototypes': prototype_lib.prototypes,
            'prototype_risks': prototype_lib.prototype_risks,
            'prototype_risk_dists': prototype_lib.prototype_risk_dists,
            'n_prototypes': prototype_lib.n_prototypes
        }, f)

    np.save(f'{data_dir}/prototype_labels.npy', proto_labels)
    np.save(f'{data_dir}/prototype_centers.npy', prototype_lib.prototypes)

    print(f"\nStage 2 completed! {n_prototypes} prototypes created.")

    return prototype_lib, proto_labels


def run_stage3(epochs=100, batch_size=16, device='cuda'):
    """运行Stage 3：条件扩散模型"""
    print("\n" + "="*70)
    print("EP-STD Stage 3: Conditional Diffusion Model")
    print("="*70)

    from epstd_stage1 import EnvironmentEncoder
    from epstd_stage2 import PrototypeLibrary
    from epstd_stage3 import (
        ConditionalRiskDiffusion, DiffusionScheduler,
        train_diffusion_model, evaluate_epstd
    )

    # 加载数据
    data_dir = 'data/processed'
    X = np.load(f'{data_dir}/X.npy')
    Y = np.load(f'{data_dir}/Y.npy')

    # 加载编码器和原型库
    env_encoder = EnvironmentEncoder(input_dim=24, output_dim=64).to(device)
    env_encoder.load_state_dict(torch.load('checkpoints/env_encoder_best.pt', map_location=device))
    env_encoder.eval()

    import pickle
    with open('checkpoints/prototype_library.pkl', 'rb') as f:
        proto_data = pickle.load(f)

    prototype_library = PrototypeLibrary(n_prototypes=proto_data['n_prototypes'])
    prototype_library.prototypes = proto_data['prototypes']
    prototype_library.prototype_risks = proto_data['prototype_risks']
    prototype_library.prototype_risk_dists = proto_data['prototype_risk_dists']

    # 划分数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # 训练扩散模型
    model, scheduler, y_mean, y_std = train_diffusion_model(
        env_encoder, prototype_library, X_train, Y_train,
        epochs=epochs, batch_size=batch_size, device=device
    )

    # 评估
    checkpoint = torch.load('checkpoints/epstd_diffusion_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])

    results = evaluate_epstd(model, scheduler, env_encoder, prototype_library, X_test, Y_test,
                            y_mean=y_mean, y_std=y_std, device=device)

    # 保存结果
    np.save(f'{data_dir}/epstd_predictions.npy', results['predictions'])
    np.save(f'{data_dir}/epstd_uncertainty.npy', results['uncertainty'])
    np.save(f'{data_dir}/epstd_zero_prob.npy', results['zero_prob'])

    print("\nStage 3 completed!")

    return model, results


def compare_with_baselines(device='cuda'):
    """与基线模型对比"""
    print("\n" + "="*70)
    print("Comparing EP-STD with Baselines")
    print("="*70)

    from src.train_stgcn_trans import calculate_metrics

    data_dir = 'data/processed'
    Y = np.load(f'{data_dir}/Y.npy')
    y_true = Y[-1]

    # 加载EP-STD预测
    epstd_pred = np.load(f'{data_dir}/epstd_predictions.npy')
    epstd_unc = np.load(f'{data_dir}/epstd_uncertainty.npy')

    # 计算指标
    epstd_metrics = calculate_metrics(epstd_pred, y_true)

    print("\nEP-STD Results:")
    print(f"  PAI@10%: {epstd_metrics.get('pai_top10', 0):.4f}")
    print(f"  PEI@10%: {epstd_metrics.get('pei_top10', 0):.4f}")
    print(f"  Recall@10%: {epstd_metrics.get('recall_top10', 0):.4f}")
    print(f"  Mean Uncertainty: {epstd_unc.mean():.4f}")

    # 尝试加载基线结果
    baseline_metrics = {}
    for baseline in ['stgcn', 'stgcn_trans', 'acrst']:
        pred_path = f'{data_dir}/{baseline}_predictions.npy'
        if os.path.exists(pred_path):
            baseline_pred = np.load(pred_path)
            baseline_metrics[baseline] = calculate_metrics(baseline_pred, y_true)
            print(f"\n{baseline.upper()} PAI@10%: {baseline_metrics[baseline].get('pai_top10', 0):.4f}")

    return epstd_metrics, baseline_metrics


def main():
    parser = argparse.ArgumentParser(description='EP-STD Full Pipeline')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', '1', '2', '3', '1+2', '2+3'],
                        help='Which stages to run')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for each stage')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for Stage 3')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with baselines after training')

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.eval_only:
        compare_with_baselines(device)
        return

    # 运行指定阶段
    if args.stage == 'all' or args.stage == '1':
        run_stage1(epochs=args.epochs, device=device)

    if args.stage == 'all' or args.stage == '2' or args.stage == '1+2':
        run_stage2(device=device)

    if args.stage == 'all' or args.stage == '3' or args.stage == '2+3':
        run_stage3(epochs=args.epochs, batch_size=args.batch_size, device=device)

    if args.compare:
        compare_with_baselines(device)

    print("\n" + "="*70)
    print("EP-STD Pipeline Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
