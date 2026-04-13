"""
每日预测生成脚本 - 使用训练好的模型生成未来7天的预测
训练一次，每日自动推理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import json
from datetime import datetime, timedelta
from train_stgcn_trans import SpatioTemporalTransformer


def load_trained_model(device='cpu'):
    """加载训练好的模型"""
    print("Loading trained model...")

    # 模型配置（应与训练时一致）
    config = {
        'static_dim': 24,
        'dynamic_dim': 7,  # 根据实际数据调整
        'semantic_dim': 1024,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_temporal_layers': 3,
        'num_spatial_layers': 2,
        'dropout': 0.1,
        'num_nodes': 1246,
        'use_semantic_gate': True,
        'use_near_repeat': True
    }

    model = SpatioTemporalTransformer(**config).to(device)

    # 加载权重
    checkpoint_path = os.path.join('..', 'checkpoints', 'best_model_trans.pt')
    if not os.path.exists(checkpoint_path):
        # 尝试相对web_vis的路径
        checkpoint_path = os.path.join('..', '..', 'checkpoints', 'best_model_trans.pt')

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model (for demo only)")

    model.eval()
    return model


def generate_predictions(days=7, use_demo_data=True):
    """
    生成未来多日的预测

    Args:
        days: 预测天数
        use_demo_data: 是否使用演示数据（如果没有真实数据）

    Returns:
        predictions: 预测结果字典
    """
    device = torch.device('cpu')  # 推理可以使用CPU

    # 尝试加载模型
    try:
        model = load_trained_model(device)
        model_available = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using demo mode (random predictions)")
        model_available = False

    # 尝试加载真实数据
    data_dir = os.path.join('..', 'data', 'processed')
    if not os.path.exists(data_dir):
        data_dir = os.path.join('..', '..', 'data', 'processed')

    predictions = {}
    n_grids = 1246

    if model_available and os.path.exists(os.path.join(data_dir, 'X.npy')):
        # 使用真实模型和数据进行推理
        print("Using real model and data for inference...")

        X = np.load(os.path.join(data_dir, 'X.npy'))
        A_crime = np.load(os.path.join(data_dir, 'adj_crime_dynamic_gaussian.npy'))
        OD = np.load(os.path.join(data_dir, 'dynamic_od_flow_1246.npy'))
        A_spatial = np.load(os.path.join(data_dir, 'adj_adaptive.npy'))
        A_distance = np.load(os.path.join(data_dir, 'adj_distance.npy'))
        A_hypergraph = np.load(os.path.join(data_dir, 'adj_hypergraph.npy'))

        # 加载语义嵌入
        semantic_embed = None
        sem_path = os.path.join(data_dir, 'semantic_embedding_rag.npy')
        if os.path.exists(sem_path):
            semantic_embed = torch.tensor(
                np.load(sem_path), dtype=torch.float32
            ).to(device)

        # 转换tensor
        A_spatial_t = torch.tensor(A_spatial, dtype=torch.float32).to(device)
        A_distance_t = torch.tensor(A_distance, dtype=torch.float32).to(device)
        A_hyper_t = torch.tensor(A_hypergraph, dtype=torch.float32).to(device)

        # 使用最后30天作为历史，预测未来
        with torch.no_grad():
            for day in range(days):
                # 这里简化处理：使用测试集的一部分作为示例
                # 实际应用中应该加载最新的实时数据
                idx = len(X) - days + day
                if idx < 30:
                    continue

                X_input = torch.tensor(
                    X[idx-30:idx], dtype=torch.float32
                ).unsqueeze(0).to(device)

                A_crime_input = torch.tensor(
                    A_crime[idx], dtype=torch.float32
                ).unsqueeze(0).to(device)

                OD_input = torch.tensor(
                    np.log1p(OD[idx]), dtype=torch.float32
                ).unsqueeze(0).to(device)

                # 提取犯罪历史
                crime_history = X_input[:, :, :, -7:]
                crime_history = crime_history[:, :, :, 0]

                # 预测
                pi, mu, theta, _, _ = model(
                    X_input, A_spatial_t, A_distance_t,
                    A_crime_input, A_hyper_t, OD_input,
                    semantic_embed=semantic_embed,
                    crime_history=crime_history
                )

                pred = ((1 - pi) * mu).squeeze().cpu().numpy()

                # 生成日期
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
                    "top_10_percent": np.argsort(pred)[-125:].tolist()
                }

                print(f"  {future_date}: Total expected = {pred.sum():.2f}")

    else:
        # 使用演示数据
        print("Using demo predictions...")
        np.random.seed(42)

        # 基础风险分布（南部和西部风险高）
        base_risk = np.random.gamma(0.5, 0.5, n_grids)

        # 添加热点
        hotspots = [586, 234, 891, 445, 723, 156, 998, 334, 667, 112]
        for h in hotspots:
            if h < n_grids:
                base_risk[h] *= 3

        for day in range(days):
            future_date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")

            daily_var = np.random.normal(1, 0.15, n_grids)
            weekend_factor = 1.3 if day >= 5 else 1.0

            risk_scores = base_risk * daily_var * weekend_factor
            risk_scores = np.clip(risk_scores, 0, 5)

            risk_levels = []
            for r in risk_scores:
                if r > 2.0:
                    risk_levels.append("very_high")
                elif r > 1.0:
                    risk_levels.append("high")
                elif r > 0.5:
                    risk_levels.append("medium")
                else:
                    risk_levels.append("low")

            predictions[future_date] = {
                "risk_scores": risk_scores.tolist(),
                "risk_levels": risk_levels,
                "expected_crimes": risk_scores.tolist(),
                "top_10_percent": np.argsort(risk_scores)[-125:].tolist()
            }

    return predictions


def save_predictions(predictions):
    """保存预测结果"""
    output_dir = os.path.join('static', 'data')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to {output_path}")
    print(f"Total days: {len(predictions)}")


if __name__ == "__main__":
    print("="*60)
    print("Daily Prediction Generator")
    print("="*60)

    predictions = generate_predictions(days=7)
    save_predictions(predictions)

    print("\nPrediction Summary:")
    for date, data in predictions.items():
        total = sum(data["risk_scores"])
        vh = data["risk_levels"].count("very_high")
        h = data["risk_levels"].count("high")
        print(f"  {date}: Total={total:.2f}, VeryHigh={vh}, High={h}")
