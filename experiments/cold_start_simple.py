"""
冷启动能力诊断实验（简化版，避免sklearn/scipy依赖）
验证"环境信号"是否被历史数据掩盖
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data():
    """加载数据（简化版，不加载模型）"""
    # 使用相对于脚本的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'processed')

    X = np.load(f"{data_dir}/X.npy")  # (samples, N, F)
    Y = np.load(f"{data_dir}/Y.npy")  # (samples, N)

    # 取最后时刻的静态特征
    static_features = X[-1, :, :24]  # (N, 24)

    # 取平均犯罪率作为风险标签
    risk_labels = Y.mean(axis=0)  # (N,)

    # 历史犯罪（用于判断"历史为0"）
    history_crime = Y[-1, :]  # (N,)

    print(f"数据加载完成:")
    print(f"  静态特征: {static_features.shape}")
    print(f"  风险标签: {risk_labels.shape}")
    print(f"  历史犯罪: {history_crime.shape}")
    print(f"  历史犯罪范围: [{history_crime.min():.4f}, {history_crime.max():.4f}]")

    return static_features, risk_labels, history_crime


def experiment_1_env_risk_correlation(static_features, risk_labels, history_crime):
    """
    实验1：环境特征与风险的相关性分析
    检查哪些环境特征与犯罪风险相关
    """
    print("\n" + "="*70)
    print("实验1：环境特征与风险的相关性")
    print("="*70)

    feature_names = ['商业POI', '交通POI', '公共POI', '道路密度', '住宅',
                     '商业用地', '工业用地', '绿化率', '夜间照明', '摄像头',
                     'CPTED监护', 'CPTED入口', 'CPTED领域', 'CPTED目标强化',
                     '建筑密度', '人口密度'] + [f'特征{i}' for i in range(16, 24)]

    correlations = []
    for i in range(min(16, static_features.shape[1])):
        corr = np.corrcoef(static_features[:, i], risk_labels)[0, 1]
        correlations.append((feature_names[i], corr))
        print(f"{feature_names[i]:15s}: r = {corr:6.3f}")

    # 找出强相关特征
    strong_corr = [c for c in correlations if abs(c[1]) > 0.2]
    print(f"\n强相关特征 (|r| > 0.2): {len(strong_corr)}个")
    for name, corr in strong_corr:
        print(f"  {name}: {corr:.3f}")

    if len(strong_corr) >= 3:
        print("[PASS] 环境特征与风险有较强相关性")
    else:
        print("[WARN] 环境特征与风险相关性较弱")

    return correlations


def experiment_2_cold_start_detection(static_features, risk_labels, history_crime):
    """
    实验2：检测"环境高风险但历史为0"的网格
    """
    print("\n" + "="*70)
    print("实验2：冷启动高风险网格检测")
    print("="*70)

    # 计算环境风险评分
    env_risk_score = (
        static_features[:, 0] * 0.3 +      # 商业POI
        static_features[:, 3] * 0.2 +      # 道路密度
        (1 - static_features[:, 8]) * 0.3 + # 低照明
        (1 - static_features[:, 9]) * 0.2   # 低监控
    )

    # 定义阈值
    high_env_threshold = np.percentile(env_risk_score, 80)
    zero_history_threshold = 0.01

    # 找环境高风险但历史为0的网格
    cold_start_mask = (env_risk_score > high_env_threshold) & (history_crime < zero_history_threshold)
    cold_start_indices = np.where(cold_start_mask)[0]

    print(f"环境高风险但历史为0的网格: {len(cold_start_indices)}个 ({len(cold_start_indices)/len(env_risk_score)*100:.1f}%)")

    if len(cold_start_indices) == 0:
        print("[WARN]  未找到冷启动网格")
        return None

    # 这些网格的真实风险
    cold_start_risks = risk_labels[cold_start_indices]

    print(f"\n冷启动网格的真实风险统计:")
    print(f"  平均风险: {cold_start_risks.mean():.4f}")
    print(f"  中位数: {np.median(cold_start_risks):.4f}")
    print(f"  标准差: {cold_start_risks.std():.4f}")
    print(f"  最大风险: {cold_start_risks.max():.4f}")

    # 与全体对比
    all_mean = risk_labels.mean()
    print(f"\n与全体网格对比:")
    print(f"  全体平均风险: {all_mean:.4f}")
    print(f"  冷启动网格平均: {cold_start_risks.mean():.4f}")
    print(f"  差异: {cold_start_risks.mean() - all_mean:.4f}")

    if cold_start_risks.mean() > all_mean * 1.2:
        print("[PASS] 冷启动网格的实际风险显著高于平均，环境预测有价值")
    elif cold_start_risks.mean() > all_mean:
        print("[WARN] 冷启动网格风险略高于平均，环境预测有一定价值")
    else:
        print("[FAIL] 冷启动网格风险不突出，环境预测价值有限")

    return cold_start_indices, cold_start_risks


def experiment_3_similar_env_analysis(static_features, risk_labels, history_crime, similarity_threshold=0.9):
    """
    实验3：环境相似网格分析
    检查环境相似的网格是否具有相似风险
    """
    print("\n" + "="*70)
    print("实验3：环境相似网格的风险一致性")
    print("="*70)

    # 归一化特征
    static_norm = static_features / (np.linalg.norm(static_features, axis=1, keepdims=True) + 1e-8)

    # 计算余弦相似度矩阵（简化：只采样部分网格）
    n_sample = min(500, len(static_features))  # 限制计算量
    sample_idx = np.random.choice(len(static_features), n_sample, replace=False)

    sample_features = static_norm[sample_idx]
    sample_risks = risk_labels[sample_idx]
    sample_history = history_crime[sample_idx]

    # 计算相似度矩阵
    similarity = sample_features @ sample_features.T

    # 找高相似度对
    pairs = []
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            if similarity[i, j] > similarity_threshold:
                pairs.append((i, j, similarity[i, j]))

    print(f"采样 {n_sample} 个网格，找到 {len(pairs)} 对高相似网格 (相似度 > {similarity_threshold})")

    if len(pairs) == 0:
        print("[WARN]  未找到相似网格对，降低阈值")
        return None

    # 计算相似网格的风险差异
    risk_diffs = []
    history_diffs = []

    for i, j, sim in pairs:
        risk_diff = abs(sample_risks[i] - sample_risks[j])
        hist_diff = abs(sample_history[i] - sample_history[j])
        risk_diffs.append(risk_diff)
        history_diffs.append(hist_diff)

    avg_risk_diff = np.mean(risk_diffs)
    avg_history_diff = np.mean(history_diffs)

    print(f"\n高相似网格对的平均差异:")
    print(f"  风险差异: {avg_risk_diff:.4f}")
    print(f"  历史犯罪差异: {avg_history_diff:.4f}")

    # 如果风险差异小但历史差异大，说明环境比历史更能预测
    if avg_risk_diff < 0.5 and avg_history_diff > 0.1:
        print("[PASS] 环境相似网格风险一致性强，环境是可靠预测因子")
    elif avg_risk_diff < 1.0:
        print("[WARN] 环境相似网格风险有一定一致性")
    else:
        print("[FAIL] 环境相似网格风险差异大，单纯环境不足以预测")

    return pairs, risk_diffs


def main():
    """主验证流程"""
    print("="*70)
    print("冷启动能力诊断实验（简化版）")
    print("验证环境信号是否被历史数据掩盖")
    print("="*70)

    # 加载数据
    try:
        static_features, risk_labels, history_crime = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 运行三个实验
    results = {}

    results['exp1'] = experiment_1_env_risk_correlation(static_features, risk_labels, history_crime)

    results['exp2'] = experiment_2_cold_start_detection(static_features, risk_labels, history_crime)

    results['exp3'] = experiment_3_similar_env_analysis(static_features, risk_labels, history_crime)

    # 综合评估
    print("\n" + "="*70)
    print("综合评估")
    print("="*70)

    exp1_pass = len([c for c in results['exp1'] if abs(c[1]) > 0.2]) >= 3
    exp2_pass = results['exp2'] is not None and results['exp2'][1].mean() > risk_labels.mean()
    exp3_pass = results['exp3'] is not None and np.mean(results['exp3'][1]) < 0.5

    print(f"实验1 (环境-风险相关): {'[PASS] 通过' if exp1_pass else '[FAIL] 未通过'}")
    print(f"实验2 (冷启动网格识别): {'[PASS] 通过' if exp2_pass else '[FAIL] 未通过'}")
    print(f"实验3 (环境相似一致性): {'[PASS] 通过' if exp3_pass else '[FAIL] 未通过'}")

    total_pass = sum([exp1_pass, exp2_pass, exp3_pass])
    print(f"\n通过测试: {total_pass}/3")

    if total_pass >= 2:
        print("\n[RESULT] 结论: 环境信号强，冷启动预测可行")
        print("   建议: 可采用EP-STD架构强化冷启动能力")
    else:
        print("\n[WARN]  结论: 环境信号弱，需重新思考冷启动策略")
        print("   建议: 可能历史数据仍是主导因素")

    print("="*70)


if __name__ == "__main__":
    main()
