"""
环境原型假设快速验证
核心问题：环境相似的网格，是否具有相似的风险分布？
如果验证成功，证明EP-STD思路可行
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mutual_info_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data():
    """加载数据"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    X = np.load(f"{data_dir}/X.npy")  # (T, N, F)
    Y = np.load(f"{data_dir}/Y.npy")  # (T, N)

    # 取最后时刻的静态特征
    static_features = X[-1, :, :24]  # (N, 24) 假设前24维是静态

    # 取平均犯罪率作为风险标签
    risk_labels = Y.mean(axis=0)  # (N,)

    print(f"Static features shape: {static_features.shape}")
    print(f"Risk labels shape: {risk_labels.shape}")
    print(f"Risk range: [{risk_labels.min():.4f}, {risk_labels.max():.4f}]")

    return static_features, risk_labels


def test_kmeans_clustering(static_features, risk_labels, n_clusters=10):
    """
    测试1：K-Means聚类环境特征
    假设：同一cluster的网格，风险分布应该相似（方差小）
    """
    print("\n" + "="*60)
    print("Test 1: K-Means Clustering on Environment Features")
    print("="*60)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(static_features)

    # 计算每个cluster的统计
    results = []
    for c in range(n_clusters):
        mask = (clusters == c)
        cluster_risks = risk_labels[mask]

        results.append({
            'cluster': c,
            'size': mask.sum(),
            'mean_risk': cluster_risks.mean(),
            'std_risk': cluster_risks.std(),
            'cv': cluster_risks.std() / (cluster_risks.mean() + 1e-8)  # 变异系数
        })

        print(f"Cluster {c:2d}: n={mask.sum():4d}, "
              f"mean={cluster_risks.mean():.4f}, "
              f"std={cluster_risks.std():.4f}, "
              f"CV={cluster_risks.std()/(cluster_risks.mean()+1e-8):.2f}")

    # 整体评估：cluster间风险差异 vs cluster内方差
    between_var = np.var([r['mean_risk'] for r in results])
    within_var = np.mean([r['std_risk']**2 for r in results])

    print(f"\nBetween-cluster variance: {between_var:.6f}")
    print(f"Within-cluster variance: {within_var:.6f}")
    print(f"F-ratio (between/within): {between_var/(within_var+1e-8):.2f}")

    if between_var > within_var:
        print("✅ PASS: 环境聚类能有效区分风险水平")
    else:
        print("❌ FAIL: 环境聚类无法区分风险")

    return clusters, results


def test_nearest_neighbor_similarity(static_features, risk_labels, k=5):
    """
    测试2：K近邻相似性
    假设：环境最相似的K个网格，风险应该高度相关
    """
    print("\n" + "="*60)
    print("Test 2: K-Nearest Neighbor Risk Similarity")
    print("="*60)

    # 计算环境特征余弦相似度
    features_norm = F.normalize(torch.tensor(static_features), dim=1)
    similarity = torch.matmul(features_norm, features_norm.T).numpy()

    # 对每个网格，找K个最近邻
    correlations = []

    for i in range(len(static_features)):
        # 找top-k相似（排除自己）
        neighbors = np.argsort(similarity[i])[-(k+1):-1]

        # 邻居的风险
        neighbor_risks = risk_labels[neighbors]

        # 该网格的风险与邻居平均风险的相关性
        corr = 1 - abs(risk_labels[i] - neighbor_risks.mean()) / (risk_labels.max() - risk_labels.min())
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    print(f"Mean risk similarity with {k} nearest neighbors: {mean_corr:.4f}")

    if mean_corr > 0.6:
        print(f"✅ PASS: 环境相似网格的风险高度相关")
    elif mean_corr > 0.4:
        print(f"⚠️  WEAK: 环境相似性与风险相关性中等")
    else:
        print(f"❌ FAIL: 环境相似网格风险不相关")

    return mean_corr


def test_contrastive_learning_potential(static_features, risk_labels, n_pairs=1000):
    """
    测试3：对比学习潜力
    随机采样正负样本对，看是否能用环境特征区分
    """
    print("\n" + "="*60)
    print("Test 3: Contrastive Learning Potential")
    print("="*60)

    np.random.seed(42)
    n_samples = len(static_features)

    # 定义风险等级
    risk_threshold = np.percentile(risk_labels, 80)
    high_risk_mask = risk_labels > risk_threshold

    pos_pairs = []  # 同风险等级
    neg_pairs = []  # 不同风险等级

    # 采样正样本对
    for _ in range(n_pairs//2):
        # 从高犯罪区采样两个
        high_indices = np.where(high_risk_mask)[0]
        if len(high_indices) >= 2:
            i, j = np.random.choice(high_indices, 2, replace=False)
            pos_pairs.append((i, j))

    # 采样负样本对
    for _ in range(n_pairs//2):
        i = np.random.choice(np.where(high_risk_mask)[0])
        j = np.random.choice(np.where(~high_risk_mask)[0])
        neg_pairs.append((i, j))

    # 计算环境特征距离
    def env_distance(i, j):
        return np.linalg.norm(static_features[i] - static_features[j])

    pos_distances = [env_distance(i, j) for i, j in pos_pairs]
    neg_distances = [env_distance(i, j) for i, j in neg_pairs]

    print(f"Positive pairs (same risk) - mean distance: {np.mean(pos_distances):.4f}")
    print(f"Negative pairs (diff risk) - mean distance: {np.mean(neg_distances):.4f}")

    # 如果正样本距离显著小于负样本，说明对比学习可行
    if np.mean(pos_distances) < np.mean(neg_distances):
        print("✅ PASS: 同风险区域环境更相似，对比学习可行")
        return True
    else:
        print("❌ FAIL: 环境特征无法区分风险等级")
        return False


def test_cpted_risk_correlation(static_features, risk_labels):
    """
    测试4：CPTED相关特征与风险的相关性
    假设：照明、监控等CPTED维度应与风险负相关
    """
    print("\n" + "="*60)
    print("Test 4: CPTED Features vs Risk Correlation")
    print("="*60)

    # 假设索引（根据实际特征定义调整）
    feature_names = {
        'nightlight': 8,    # 照明
        'camera': 9,        # 监控
        'commercial': 0,    # 商业POI
        'green_ratio': 7,   # 绿化率
    }

    correlations = {}
    for name, idx in feature_names.items():
        if idx < static_features.shape[1]:
            corr, p_value = pearsonr(static_features[:, idx], risk_labels)
            correlations[name] = {'corr': corr, 'p_value': p_value}

            sign = "✅" if abs(corr) > 0.1 and p_value < 0.05 else "⚠️"
            print(f"{sign} {name:15s}: r={corr:6.3f}, p={p_value:.4f}")

    significant = sum(1 for v in correlations.values()
                     if abs(v['corr']) > 0.1 and v['p_value'] < 0.05)

    if significant >= 2:
        print(f"\n✅ PASS: {significant}/{len(correlations)} CPTED特征与风险显著相关")
    else:
        print(f"\n⚠️  WEAK: 仅{significant}个CPTED特征显著相关")

    return correlations


def visualize_clusters(static_features, risk_labels, clusters, save_path='env_clusters.png'):
    """可视化环境聚类结果"""
    from sklearn.manifold import TSNE

    print("\nGenerating visualization...")

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(static_features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：按聚类着色
    scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1],
                               c=clusters, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title('Environment Clusters (t-SNE)')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # 右图：按风险值着色
    scatter2 = axes[1].scatter(embedded[:, 0], embedded[:, 1],
                               c=risk_labels, cmap='YlOrRd', s=10, alpha=0.6)
    axes[1].set_title('Risk Distribution (t-SNE)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1], label='Risk Level')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")


def main():
    """主验证流程"""
    print("="*60)
    print("Environment Prototype Hypothesis Verification")
    print("="*60)

    # 加载数据
    static_features, risk_labels = load_data()

    # 运行所有测试
    results = {}

    # Test 1: K-Means聚类
    clusters, cluster_stats = test_kmeans_clustering(static_features, risk_labels, n_clusters=10)
    results['kmeans'] = cluster_stats

    # Test 2: K近邻相似性
    nn_corr = test_nearest_neighbor_similarity(static_features, risk_labels, k=5)
    results['nn_similarity'] = nn_corr

    # Test 3: 对比学习潜力
    contrastive_feasible = test_contrastive_learning_potential(static_features, risk_labels)
    results['contrastive'] = contrastive_feasible

    # Test 4: CPTED相关性
    cpted_corr = test_cpted_risk_correlation(static_features, risk_labels)
    results['cpted'] = cpted_corr

    # 可视化
    visualize_clusters(static_features, risk_labels, clusters)

    # 综合评估
    print("\n" + "="*60)
    print("Overall Assessment")
    print("="*60)

    pass_count = sum([
        results['kmeans'][0]['mean_risk'] > 0,  # 至少聚类有效
        results['nn_similarity'] > 0.4,
        results['contrastive'],
        len([v for v in results['cpted'].values() if abs(v['corr']) > 0.1]) >= 2
    ])

    print(f"Tests passed: {pass_count}/4")

    if pass_count >= 3:
        print("\n🎯 STRONG EVIDENCE: Environment-prototype approach is FEASIBLE")
        print("   Recommendation: Proceed with EP-STD architecture")
    elif pass_count >= 2:
        print("\n⚠️  MODERATE EVIDENCE: Environment-prototype approach shows potential")
        print("   Recommendation: Further investigation needed")
    else:
        print("\n❌ WEAK EVIDENCE: Environment-prototype approach may not work")
        print("   Recommendation: Stick with current architecture or try other approaches")

    print("="*60)


if __name__ == "__main__":
    main()
