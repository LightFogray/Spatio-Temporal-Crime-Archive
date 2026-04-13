"""
简化版数据准备脚本 - 不使用geopandas，直接用numpy生成示例数据
当shapefile读取有问题时使用此脚本
"""

import numpy as np
import json
import os


def generate_sample_data():
    """生成示例网格和预测数据"""
    print("Generating sample data for visualization...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "static", "data")
    os.makedirs(data_dir, exist_ok=True)

    # 模拟1246个网格（芝加哥实际网格数）
    n_grids = 1246

    # 生成网格中心坐标（模拟芝加哥区域）
    # 芝加哥大致范围: lat 41.64-42.02, lon -87.91--87.52
    np.random.seed(42)

    # 创建更真实的空间分布（南部和西部风险更高）
    base_lats = np.random.uniform(41.65, 42.00, n_grids)
    base_lons = np.random.uniform(-87.90, -87.55, n_grids)

    # 基础风险（南部和西部风险更高）
    base_risk = np.zeros(n_grids)
    for i in range(n_grids):
        # 南部（纬度低）风险高
        south_factor = (41.85 - base_lats[i]) / 0.2  # 南部更高
        # 西部（经度小）风险高
        west_factor = (-87.55 - base_lons[i]) / 0.35  # 西部更高

        base_risk[i] = max(0, south_factor * 0.5 + west_factor * 0.3 + np.random.gamma(0.3, 0.5))

    # 添加一些明确的热点区域
    hotspots = [586, 234, 891, 445, 723, 156, 998, 334, 667, 112]
    for h in hotspots:
        if h < n_grids:
            base_risk[h] += 3.0

    # 生成7天预测数据
    from datetime import datetime, timedelta
    predictions = {}

    for day in range(7):
        date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")

        # 每天的风险有所变化
        daily_variation = np.random.normal(1, 0.15, n_grids)
        weekend_factor = 1.3 if day >= 5 else 1.0  # 周末风险更高

        risk_scores = base_risk * daily_variation * weekend_factor
        risk_scores = np.clip(risk_scores, 0, 5)

        # 分级
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

        predictions[date] = {
            "risk_scores": risk_scores.tolist(),
            "risk_levels": risk_levels,
            "expected_crimes": risk_scores.tolist(),
            "top_10_percent": np.argsort(risk_scores)[-int(n_grids*0.1):].tolist()
        }

    # 保存预测数据
    pred_path = os.path.join(data_dir, "predictions.json")
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {pred_path}")

    # 生成网格元数据
    metadata = []
    for i in range(n_grids):
        metadata.append({
            "grid_id": i,
            "center_lat": float(base_lats[i]),
            "center_lon": float(base_lons[i]),
            "area": 0.25  # 约500m x 500m = 0.25 km²
        })

    meta_path = os.path.join(data_dir, "grid_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {meta_path}")

    # 生成简化的GeoJSON（网格用点表示，实际用多边形更好）
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for i in range(n_grids):
        feature = {
            "type": "Feature",
            "properties": {
                "grid_id": i
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [base_lons[i] - 0.002, base_lats[i] - 0.002],
                    [base_lons[i] + 0.002, base_lats[i] - 0.002],
                    [base_lons[i] + 0.002, base_lats[i] + 0.002],
                    [base_lons[i] - 0.002, base_lats[i] + 0.002],
                    [base_lons[i] - 0.002, base_lats[i] - 0.002]
                ]]
            }
        }
        geojson["features"].append(feature)

    grid_path = os.path.join(data_dir, "chicago_grid.geojson")
    with open(grid_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
    print(f"Saved grid GeoJSON to {grid_path}")

    print(f"\nGenerated {n_grids} grid cells with {len(predictions)} days of predictions")
    return True


if __name__ == "__main__":
    print("="*60)
    print("Simplified Data Preparation (No GeoPandas Required)")
    print("="*60)

    success = generate_sample_data()

    if success:
        print("\n" + "="*60)
        print("Data preparation complete!")
        print("="*60)
        print("\nNext step: Run 'python app.py' to start the web server")
    else:
        print("\nData preparation failed!")
