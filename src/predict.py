"""
模型推理脚本 - 使用训练好的模型进行未来多日的暴力犯罪预测
无需重新训练，直接加载模型权重进行前向传播
"""

import torch
import numpy as np
import json
import os
from datetime import datetime, timedelta
from train_stgcn_trans import (
    SpatioTemporalTransformer, CrimeDataset,
    zinb_loss, calculate_advanced_metrics
)
from torch.utils.data import DataLoader


def load_model_for_inference(checkpoint_path, config, device):
    """
    加载训练好的模型进行推理

    Args:
        checkpoint_path: 模型检查点路径 (.pt文件)
        config: 模型配置参数
        device: 计算设备

    Returns:
        model: 加载好的模型
    """
    print(f"Loading model from {checkpoint_path}...")

    model = SpatioTemporalTransformer(
        static_dim=config['static_dim'],
        dynamic_dim=config['dynamic_dim'],
        semantic_dim=config['semantic_dim'],
        hidden_dim=config.get('hidden_dim', 64),
        num_heads=config.get('num_heads', 4),
        num_temporal_layers=config.get('num_temporal_layers', 3),
        num_spatial_layers=config.get('num_spatial_layers', 2),
        dropout=config.get('dropout', 0.1),
        num_nodes=config['num_nodes'],
        use_semantic_gate=config.get('use_semantic_gate', True),
        use_near_repeat=config.get('use_near_repeat', True),
        distance_matrix=None
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # 设置为评估模式

    print("Model loaded successfully!")
    return model


def prepare_inference_data(X, A_crime, OD, window_size=30, static_dim=24):
    """
    准备推理数据

    Args:
        X: 完整的特征数据 (T, N, F)
        A_crime: 动态犯罪图 (T, N, N)
        OD: OD流特征 (T, N, 4)
        window_size: 历史窗口大小
        static_dim: 静态特征维度

    Returns:
        X_window: 窗口化的输入数据
        A_window: 窗口化的犯罪图
        OD_window: 窗口化的OD数据
        dates: 对应的预测日期索引
    """
    T, N, F = X.shape

    X_windows = []
    A_windows = []
    OD_windows = []
    date_indices = []

    # 使用最后window_size天作为历史，预测接下来的每一天
    for i in range(window_size, T):
        X_windows.append(X[i-window_size:i])
        A_windows.append(A_crime[i])
        OD_windows.append(OD[i])
        date_indices.append(i)

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(A_windows, dtype=np.float32),
        np.array(OD_windows, dtype=np.float32),
        date_indices
    )


def predict_future_days(model, X_recent, A_recent, OD_recent,
                       A_spatial, A_distance, A_hypergraph,
                       semantic_embed, device, days=7):
    """
    预测未来多天的犯罪风险

    Args:
        model: 加载好的模型
        X_recent: 最近的历史数据 (T_recent, N, F)
        A_recent: 最近的犯罪图 (T_recent, N, N)
        OD_recent: 最近的OD数据 (T_recent, N, 4)
        days: 预测未来天数

    Returns:
        predictions: 每天的预测结果字典
    """
    print(f"Predicting next {days} days...")

    model.eval()
    predictions = {}

    with torch.no_grad():
        for day in range(days):
            # 准备输入数据
            X_input = torch.tensor(X_recent[-30:], dtype=torch.float32).unsqueeze(0).to(device)
            A_crime_input = torch.tensor(A_recent[-1], dtype=torch.float32).unsqueeze(0).to(device)
            OD_input = torch.tensor(OD_recent[-1], dtype=torch.float32).unsqueeze(0).to(device)

            # 提取犯罪历史用于近重复效应
            crime_history = X_input[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            # 前向传播
            pi, mu, theta, _, _ = model(
                X_input, A_spatial, A_distance, A_crime_input, A_hypergraph,
                OD_input, semantic_embed=semantic_embed, crime_history=crime_history
            )

            # 计算期望犯罪数
            pred = ((1 - pi) * mu).squeeze().cpu().numpy()

            # 确定日期
            future_date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")

            # 分级
            risk_levels = []
            for r in pred:
                if r > 2.0:
                    risk_levels.append("very_high")
                elif r > 1.0:
                    risk_levels.append("high")
                elif r > 0.5:
                    risk_levels.append("medium")
                else:
                    risk_levels.append("low")

            predictions[future_date] = {
                "risk_scores": pred.tolist(),
                "risk_levels": risk_levels,
                "expected_crimes": pred.tolist(),
                "top_10_percent": np.argsort(pred)[-int(len(pred)*0.1):].tolist(),
                "pi": pi.squeeze().cpu().numpy().tolist(),
                "mu": mu.squeeze().cpu().numpy().tolist(),
                "theta": theta.squeeze().cpu().numpy().tolist()
            }

            # 模拟：用预测结果更新历史（实际应用中应使用真实观测值）
            # 这里简化处理，实际应该等待真实数据后再进行下一步预测
            print(f"  Day {day+1} ({future_date}): Expected crimes = {pred.sum():.2f}")

    return predictions


def predict_from_existing_data(model, X, A_crime, OD,
                               A_spatial, A_distance, A_hypergraph,
                               semantic_embed, device,
                               start_idx=None, days=7):
    """
    从已有数据集中预测指定日期范围的犯罪风险
    用于测试集评估或历史数据回测

    Args:
        model: 加载好的模型
        X: 完整特征数据
        A_crime: 完整犯罪图数据
        OD: 完整OD数据
        start_idx: 开始预测的索引（None表示从最后开始）
        days: 预测天数

    Returns:
        predictions: 预测结果
        ground_truth: 真实值（如果可用）
    """
    if start_idx is None:
        start_idx = len(X) - days

    print(f"Predicting from index {start_idx} to {start_idx + days}...")

    model.eval()
    predictions = {}

    with torch.no_grad():
        for i in range(days):
            idx = start_idx + i
            if idx >= len(X):
                break

            # 准备窗口数据
            if idx < 30:
                print(f"  Skipping index {idx}: insufficient history")
                continue

            X_window = X[idx-30:idx]
            A_crime_window = A_crime[idx]
            OD_window = OD[idx]

            X_input = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0).to(device)
            A_crime_input = torch.tensor(A_crime_window, dtype=torch.float32).unsqueeze(0).to(device)
            OD_input = torch.tensor(OD_window, dtype=torch.float32).unsqueeze(0).to(device)

            # 提取犯罪历史
            crime_history = X_input[:, :, :, -7:]
            crime_history = crime_history[:, :, :, 0]

            # 预测
            pi, mu, theta, _, _ = model(
                X_input, A_spatial, A_distance, A_crime_input, A_hypergraph,
                OD_input, semantic_embed=semantic_embed, crime_history=crime_history
            )

            pred = ((1 - pi) * mu).squeeze().cpu().numpy()

            # 生成日期（基于索引偏移）
            date = (datetime(2024, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%d")

            # 分级
            risk_levels = []
            for r in pred:
                if r > 2.0:
                    risk_levels.append("very_high")
                elif r > 1.0:
                    risk_levels.append("high")
                elif r > 0.5:
                    risk_levels.append("medium")
                else:
                    risk_levels.append("low")

            predictions[date] = {
                "risk_scores": pred.tolist(),
                "risk_levels": risk_levels,
                "expected_crimes": pred.tolist(),
                "top_10_percent": np.argsort(pred)[-int(len(pred)*0.1):].tolist()
            }

            print(f"  {date}: Expected total = {pred.sum():.2f}")

    return predictions


def save_predictions(predictions, output_path):
    """保存预测结果到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_path}")


def main():
    """主函数 - 演示如何使用训练好的模型进行推理"""
    print("="*60)
    print("暴力犯罪预测 - 模型推理模式")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading data...")
    X = np.load("../data/processed/X.npy")
    Y = np.load("../data/processed/Y.npy")
    OD = np.load("../data/processed/dynamic_od_flow_1246.npy")
    OD = np.log1p(OD)

    A_spatial = np.load("../data/processed/adj_adaptive.npy")
    A_distance = np.load("../data/processed/adj_distance.npy")
    A_crime_dynamic = np.load("../data/processed/adj_crime_dynamic_gaussian.npy")
    A_hypergraph = np.load("../data/processed/adj_hypergraph.npy")

    # 加载语义嵌入
    semantic_embed = None
    semantic_path = "../data/processed/semantic_embedding_rag.npy"
    if os.path.exists(semantic_path):
        semantic_embed = np.load(semantic_path)
        print(f"Loaded semantic embedding: {semantic_embed.shape}")

    # 模型配置
    static_dim = 24
    dynamic_dim = X.shape[2] - static_dim
    semantic_dim = semantic_embed.shape[1] if semantic_embed is not None else 0
    num_nodes = X.shape[1]

    config = {
        'static_dim': static_dim,
        'dynamic_dim': dynamic_dim,
        'semantic_dim': semantic_dim,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_temporal_layers': 3,
        'num_spatial_layers': 2,
        'dropout': 0.1,
        'num_nodes': num_nodes,
        'use_semantic_gate': True,
        'use_near_repeat': True
    }

    # 加载模型
    checkpoint_path = "../checkpoints/best_model_trans.pt"
    model = load_model_for_inference(checkpoint_path, config, device)

    # 转换图为tensor
    A_spatial_t = torch.tensor(A_spatial, dtype=torch.float32).to(device)
    A_distance_t = torch.tensor(A_distance, dtype=torch.float32).to(device)
    A_hypergraph_t = torch.tensor(A_hypergraph, dtype=torch.float32).to(device)

    if semantic_embed is not None:
        semantic_embed_t = torch.tensor(semantic_embed, dtype=torch.float32).to(device)
    else:
        semantic_embed_t = None

    # 使用测试集最后7天进行预测演示
    print("\n" + "="*60)
    print("Generating predictions for next 7 days...")
    print("="*60)

    # 方法1: 从已有数据预测（用于验证）
    test_start_idx = len(X) - 7
    predictions = predict_from_existing_data(
        model, X, A_crime_dynamic, OD,
        A_spatial_t, A_distance_t, A_hypergraph_t,
        semantic_embed_t, device,
        start_idx=test_start_idx, days=7
    )

    # 保存预测结果（用于Web可视化）
    output_path = "../web_vis/static/data/predictions.json"
    save_predictions(predictions, output_path)

    # 打印摘要
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    for date, pred_data in predictions.items():
        total_crimes = sum(pred_data["risk_scores"])
        very_high = pred_data["risk_levels"].count("very_high")
        high = pred_data["risk_levels"].count("high")
        print(f"{date}: Total={total_crimes:.2f}, VeryHigh={very_high}, High={high}")

    print("\n" + "="*60)
    print("Inference complete!")
    print(f"Predictions saved to: {output_path}")
    print("Run 'python app.py' in web_vis folder to visualize")
    print("="*60)


if __name__ == "__main__":
    main()
