"""
冷启动能力诊断实验
验证"环境信号"是否被历史数据掩盖
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_model_and_data():
    """加载训练好的模型和数据"""
    from src.train_stgcn_trans import SpatioTemporalTransformer, CrimeDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data_dir = os.path.join('..', 'data', 'processed')
    X = np.load(f"{data_dir}/X.npy")
    Y = np.load(f"{data_dir}/Y.npy")

    # 加载语义嵌入
    semantic_embed = None
    if os.path.exists(f"{data_dir}/semantic_embedding_rag.npy"):
        semantic_embed = np.load(f"{data_dir}/semantic_embedding_rag.npy")

    # 加载最佳模型
    model = SpatioTemporalTransformer(
        static_dim=24,
        dynamic_dim=19,
        semantic_dim=1024 if semantic_embed is not None else 0,
        hidden_dim=64,
        num_heads=4,
        num_nodes=1246,
        use_semantic_gate=True,
        use_near_repeat=True
    ).to(device)

    checkpoint_path = os.path.join('..', 'checkpoints', 'best_model_trans.pt')
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random weights")

    model.eval()

    return model, X, Y, semantic_embed, device


def experiment_1_zero_history_benchmark(model, X, Y, semantic_embed, device):
    """
    实验1：纯环境冷启动测试
    强制历史犯罪为0，仅保留环境特征，观测热点召回率
    """
    print("\n" + "="*70)
    print("实验1：纯环境冷启动测试 (Zero-History Benchmark)")
    print("="*70)

    # 取最后一个测试样本
    # X shape: (samples, N, F) - 已经是窗口后的数据
    X_sample = torch.tensor(X[-1:], dtype=torch.float32).to(device)  # (1, N, F)
    Y_true = Y[-1]  # (N,)

    # 找到真实热点 (Top 10%)
    n_hotspot = int(0.1 * len(Y_true))
    true_hotspots = np.argsort(Y_true)[-n_hotspot:]

    print(f"真实热点数量: {len(true_hotspots)}")

    # 创建纯环境输入（历史犯罪归零）
    X_env_only = X_sample.clone()

    # 假设动态特征中犯罪历史在特定位置
    # 根据实际特征定义调整这些索引
    crime_history_indices = list(range(24, 31))  # 假设 crime_lag_1d 到 crime_lag_7d

    # 将犯罪历史归零 (注意：X_sample是3维: (1, N, F))
    X_env_only[:, :, crime_history_indices] = 0

    # 预测1：使用完整输入
    with torch.no_grad():
        crime_history = X_sample[:, :, :, -7:]
        crime_history = crime_history[:, :, :, 0]

        semantic_tensor = None
        if semantic_embed is not None:
            semantic_tensor = torch.tensor(semantic_embed, dtype=torch.float32).to(device)

        # 构建图数据（简化版，实际需要完整图结构）
        N = 1246
        A_spatial = torch.eye(N).to(device)
        A_distance = torch.eye(N).to(device)
        A_crime = torch.eye(N).unsqueeze(0).to(device)
        A_hypergraph = torch.eye(N).to(device)
        OD = torch.zeros(1, N, 4).to(device)

        pi_full, mu_full, _, _, _ = model(
            X_sample, A_spatial, A_distance, A_crime,
            A_hypergraph, OD, semantic_tensor, crime_history
        )
        pred_full = ((1 - pi_full) * mu_full).squeeze().cpu().numpy()

    # 预测2：纯环境（历史归零）
    with torch.no_grad():
        crime_history_env = X_env_only[:, :, :, -7:]
        crime_history_env = crime_history_env[:, :, :, 0]

        pi_env, mu_env, _, _, _ = model(
            X_env_only, A_spatial, A_distance, A_crime,
            A_hypergraph, OD, semantic_tensor, crime_history_env
        )
        pred_env = ((1 - pi_env) * mu_env).squeeze().cpu().numpy()

    # 评估热点召回率
    def calculate_hit_rate(pred, true_hotspots, k_percent=0.1):
        n_top = int(k_percent * len(pred))
        pred_hotspots = np.argsort(pred)[-n_top:]
        hits = len(set(pred_hotspots) & set(true_hotspots))
        return hits / len(true_hotspots)

    hit_rate_full = calculate_hit_rate(pred_full, true_hotspots)
    hit_rate_env = calculate_hit_rate(pred_env, true_hotspots)

    print(f"\n完整输入的热点召回率: {hit_rate_full:.4f}")
    print(f"纯环境输入的热点召回率: {hit_rate_env:.4f}")
    print(f"历史信息贡献度: {(hit_rate_full - hit_rate_env) / hit_rate_full * 100:.1f}%")

    if hit_rate_env > 0.3:
        print("✅ 环境特征具有较强的独立预测能力")
    elif hit_rate_env > 0.1:
        print("⚠️  环境特征有一定预测能力，但依赖历史数据")
    else:
        print("❌ 环境特征几乎无法独立预测，模型过度依赖历史")

    return {
        'hit_rate_full': hit_rate_full,
        'hit_rate_env': hit_rate_env,
        'true_hotspots': true_hotspots,
        'pred_full': pred_full,
        'pred_env': pred_env
    }


def experiment_2_cross_grid_alignment(model, X, Y, semantic_embed, device, similarity_threshold=0.8):
    """
    实验2：跨时空对齐分析
    找环境相似但历史犯罪差异大的网格对，检测历史依赖偏见
    """
    print("\n" + "="*70)
    print("实验2：跨时空对齐分析 (Cross-Grid Alignment)")
    print("="*70)

    # 计算环境特征相似度（静态特征 + 语义）
    static_features = X[-1, :, :24]  # (N, 24)

    if semantic_embed is not None:
        # 拼接静态特征和语义
        env_features = np.concatenate([static_features, semantic_embed], axis=1)
    else:
        env_features = static_features

    # 归一化
    env_features_norm = env_features / (np.linalg.norm(env_features, axis=1, keepdims=True) + 1e-8)

    # 计算余弦相似度矩阵
    similarity_matrix = env_features_norm @ env_features_norm.T

    # 历史犯罪数据
    crime_history = X[:, :, 24:31].mean(axis=(0, 2))  # 平均历史犯罪

    # 找环境相似但对（一个历史高，一个历史为0）
    pairs = []
    N = len(env_features)

    for i in range(N):
        if crime_history[i] > 0.1:  # i是高犯罪
            # 找环境相似且历史为0的j
            for j in range(N):
                if crime_history[j] < 0.01 and similarity_matrix[i, j] > similarity_threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))

    print(f"找到 {len(pairs)} 对 '环境相似但历史差异大' 的网格对")

    if len(pairs) == 0:
        print("⚠️  未找到符合条件的网格对，降低相似度阈值重试")
        return None

    # 用模型预测这些网格对的风险
    X_sample = torch.tensor(X[-1:], dtype=torch.float32).to(device)

    with torch.no_grad():
        N = 1246
        A_spatial = torch.eye(N).to(device)
        A_distance = torch.eye(N).to(device)
        A_crime = torch.eye(N).unsqueeze(0).to(device)
        A_hypergraph = torch.eye(N).to(device)
        OD = torch.zeros(1, N, 4).to(device)

        semantic_tensor = None
        if semantic_embed is not None:
            semantic_tensor = torch.tensor(semantic_embed, dtype=torch.float32).to(device)

        crime_h = X_sample[:, :, :, -7:]
        crime_h = crime_h[:, :, :, 0]

        pi, mu, _, _, _ = model(
            X_sample, A_spatial, A_distance, A_crime,
            A_hypergraph, OD, semantic_tensor, crime_h
        )
        pred_risk = ((1 - pi) * mu).squeeze().cpu().numpy()

    # 分析预测差异
    risk_diffs = []
    for i, j, sim in pairs:
        diff = abs(pred_risk[i] - pred_risk[j])
        risk_diffs.append({
            'grid_high_history': i,
            'grid_zero_history': j,
            'similarity': sim,
            'risk_i': pred_risk[i],
            'risk_j': pred_risk[j],
            'risk_diff': diff,
            'history_i': crime_history[i],
            'history_j': crime_history[j]
        })

    avg_diff = np.mean([d['risk_diff'] for d in risk_diffs])
    max_diff = np.max([d['risk_diff'] for d in risk_diffs])

    print(f"\n环境相似网格对（历史差异大）的预测风险差异:")
    print(f"平均差异: {avg_diff:.4f}")
    print(f"最大差异: {max_diff:.4f}")

    # 示例输出
    print(f"\n示例网格对（Top 3差异）:")
    sorted_pairs = sorted(risk_diffs, key=lambda x: x['risk_diff'], reverse=True)[:3]
    for p in sorted_pairs:
        print(f"  网格 {p['grid_high_history']:4d} (历史={p['history_i']:.3f}) vs "
              f"网格 {p['grid_zero_history']:4d} (历史={p['history_j']:.3f}): "
              f"预测风险差异={p['risk_diff']:.3f}, 环境相似度={p['similarity']:.3f}")

    if avg_diff < 0.5:
        print("✅ 模型能正确对齐环境相似的网格，历史偏见较小")
    elif avg_diff < 1.0:
        print("⚠️  模型对环境对齐一般，存在中等程度历史偏见")
    else:
        print("❌ 模型严重依赖历史数据，忽视环境相似性")

    return risk_diffs


def experiment_3_risk_aware_sampling(model, X, Y, device):
    """
    实验3：高风险低历史样本检测
    识别"环境风险高但历史为0"的样本，检查模型是否能发现它们
    """
    print("\n" + "="*70)
    print("实验3：高风险低历史样本检测")
    print("="*70)

    # 计算环境风险评分（基于POI密度、商业度等）
    static_features = X[-1, :, :24]

    # 简单环境风险评分：商业POI + 道路密度 - 照明 - 摄像头
    env_risk_score = (
        static_features[:, 0] * 0.3 +      # 商业POI
        static_features[:, 3] * 0.2 +      # 道路密度
        (1 - static_features[:, 8]) * 0.3 + # 低照明
        (1 - static_features[:, 9]) * 0.2   # 低监控
    )

    # 历史犯罪（平均）
    history_crime = X[:, :, 24:31].mean(axis=(0, 2))

    # 找"环境高风险但历史为0"的网格
    high_env_risk_threshold = np.percentile(env_risk_score, 80)

    candidates = []
    for i in range(len(env_risk_score)):
        if env_risk_score[i] > high_env_risk_threshold and history_crime[i] < 0.01:
            candidates.append({
                'grid_id': i,
                'env_risk': env_risk_score[i],
                'history_crime': history_crime[i]
            })

    print(f"环境高风险但历史为0的候选网格: {len(candidates)}个")

    if len(candidates) == 0:
        print("未找到符合条件的网格")
        return None

    # 模型预测这些网格的风险
    X_sample = torch.tensor(X[-1:], dtype=torch.float32).to(device)

    with torch.no_grad():
        N = 1246
        A_spatial = torch.eye(N).to(device)
        A_distance = torch.eye(N).to(device)
        A_crime = torch.eye(N).unsqueeze(0).to(device)
        A_hypergraph = torch.eye(N).to(device)
        OD = torch.zeros(1, N, 4).to(device)

        crime_h = X_sample[:, :, :, -7:]
        crime_h = crime_h[:, :, :, 0]

        pi, mu, _, _, _ = model(
            X_sample, A_spatial, A_distance, A_crime,
            A_hypergraph, OD, None, crime_h
        )
        pred_risk = ((1 - pi) * mu).squeeze().cpu().numpy()

    # 检查模型是否能发现这些高风险网格
    high_risk_predictions = []
    for c in candidates:
        grid_id = c['grid_id']
        pred = pred_risk[grid_id]
        percentile = (pred_risk < pred).sum() / len(pred_risk) * 100

        high_risk_predictions.append({
            **c,
            'pred_risk': pred,
            'percentile': percentile
        })

    # 统计有多少被模型识别为高风险（Top 20%）
    recognized = [p for p in high_risk_predictions if p['percentile'] > 80]
    recognition_rate = len(recognized) / len(high_risk_predictions) * 100

    print(f"\n模型识别能力:")
    print(f"候选网格中，被预测为高风险(Top 20%)的比例: {recognition_rate:.1f}%")

    if recognition_rate > 50:
        print("✅ 模型能较好识别'高风险低历史'的冷启动网格")
    elif recognition_rate > 25:
        print("⚠️  模型能识别部分冷启动网格，但能力有限")
    else:
        print("❌ 模型几乎无法识别冷启动网格，需要优化")

    print(f"\nTop 5被正确识别的冷启动网格:")
    top_recognized = sorted(recognized, key=lambda x: x['pred_risk'], reverse=True)[:5]
    for r in top_recognized:
        print(f"  网格 {r['grid_id']:4d}: 环境风险={r['env_risk']:.3f}, "
              f"预测风险={r['pred_risk']:.3f}, 排名百分位={r['percentile']:.1f}%")

    return high_risk_predictions


def main():
    print("="*70)
    print("冷启动能力诊断实验")
    print("验证环境信号是否被历史数据掩盖")
    print("="*70)

    # 加载模型和数据
    try:
        model, X, Y, semantic_embed, device = load_model_and_data()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保模型已训练并保存 checkpoint")
        return

    # 运行三个实验
    results = {}

    results['exp1'] = experiment_1_zero_history_benchmark(
        model, X, Y, semantic_embed, device
    )

    results['exp2'] = experiment_2_cross_grid_alignment(
        model, X, Y, semantic_embed, device, similarity_threshold=0.8
    )

    results['exp3'] = experiment_3_risk_aware_sampling(
        model, X, Y, device
    )

    # 综合评估
    print("\n" + "="*70)
    print("综合评估")
    print("="*70)

    exp1_pass = results['exp1']['hit_rate_env'] > 0.3
    exp2_pass = results['exp2'] is not None and np.mean([d['risk_diff'] for d in results['exp2']]) < 0.5
    exp3_pass = len([p for p in results['exp3'] if p['percentile'] > 80]) / len(results['exp3']) > 0.5 if results['exp3'] else False

    print(f"实验1 (环境独立预测): {'✅ 通过' if exp1_pass else '❌ 未通过'}")
    print(f"实验2 (历史偏见检测): {'✅ 通过' if exp2_pass else '❌ 未通过'}")
    print(f"实验3 (冷启动识别): {'✅ 通过' if exp3_pass else '❌ 未通过'}")

    total_pass = sum([exp1_pass, exp2_pass, exp3_pass])
    print(f"\n通过测试: {total_pass}/3")

    if total_pass >= 2:
        print("\n🎯 当前模型已具备一定冷启动能力，可通过微调优化")
    else:
        print("\n⚠️  当前模型严重依赖历史数据，建议采用EP-STD架构重构")


if __name__ == "__main__":
    main()
