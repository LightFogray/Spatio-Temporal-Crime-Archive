"""
准备Web可视化所需的数据
- 将shapefile转换为GeoJSON
- 加载预测结果
- 生成示例预测数据（如果还没有真实预测结果）
"""

import geopandas as gpd
import numpy as np
import json
import os
from datetime import datetime, timedelta


def convert_grid_to_geojson():
    """将网格shapefile转换为GeoJSON格式"""
    print("Converting grid shapefile to GeoJSON...")

    # 读取网格数据 - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grid_path = os.path.join(script_dir, "..", "data", "processed", "chicago_grid.shp")
    grid_path = os.path.normpath(grid_path)

    print(f"Looking for grid file at: {grid_path}")

    if not os.path.exists(grid_path):
        print(f"Error: {grid_path} not found")
        # 尝试其他可能的路径
        alt_paths = [
            "../data/processed/chicago_grid.shp",
            "data/processed/chicago_grid.shp",
            "./data/processed/chicago_grid.shp",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Found at alternative path: {alt_path}")
                grid_path = alt_path
                break
        else:
            print("Grid file not found in any location")
            return None

    try:
        # 使用geopandas读取shapefile
        grid = gpd.read_file(grid_path)
        print(f"Loaded {len(grid)} grid cells")
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None

    # 确保使用WGS84坐标系（Web地图标准）
    if grid.crs != "EPSG:4326":
        grid = grid.to_crs("EPSG:4326")

    # 添加网格ID
    grid['grid_id'] = range(len(grid))

    # 保存为GeoJSON - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "static", "data", "chicago_grid.geojson")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        grid.to_file(output_path, driver='GeoJSON')
        print(f"Saved GeoJSON to {output_path}")
    except Exception as e:
        print(f"Error saving GeoJSON: {e}")
        # 备选：使用json直接保存
        import json
        geojson_dict = grid.__geo_interface__
        with open(output_path, 'w') as f:
            json.dump(geojson_dict, f)
        print(f"Saved GeoJSON (alternative method) to {output_path}")

    return grid


def generate_sample_predictions(grid, days=7):
    """
    生成示例预测数据（模拟未来7天的预测结果）
    如果已有真实预测结果，可以替换这部分
    """
    print("Generating sample prediction data...")

    n_grids = len(grid)
    predictions = {}

    # 基础风险分布（基于芝加哥实际犯罪分布特点）
    # 南部和西部风险较高
    base_risk = np.random.gamma(0.5, 0.5, n_grids)

    # 添加一些热点区域
    hotspots = np.random.choice(n_grids, size=50, replace=False)
    for h in hotspots:
        base_risk[h] *= 3

    for day in range(days):
        date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")

        # 每天的风险有所变化
        daily_variation = np.random.normal(1, 0.2, n_grids)
        risk_scores = base_risk * daily_variation
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

    # 保存预测数据 - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "static", "data", "predictions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved predictions to {output_path}")

    return predictions


def load_real_predictions():
    """加载真实的模型预测结果（如果存在）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(script_dir, "static", "data", "predictions.json")

    # 检查web_vis自己的数据目录
    if os.path.exists(pred_path):
        print("Loading predictions from web_vis data...")
        with open(pred_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 尝试项目主目录
    alt_path = os.path.join(script_dir, "..", "data", "processed", "predictions.json")
    alt_path = os.path.normpath(alt_path)
    if os.path.exists(alt_path):
        print("Loading real predictions from project data...")
        with open(alt_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def generate_grid_metadata(grid):
    """生成网格元数据（中心坐标等）"""
    print("Generating grid metadata...")

    metadata = []
    for idx, row in grid.iterrows():
        centroid = row.geometry.centroid
        metadata.append({
            "grid_id": idx,
            "center_lat": centroid.y,
            "center_lon": centroid.x,
            "area": row.geometry.area
        })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "static", "data", "grid_metadata.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("Preparing data for web visualization")
    print("="*60)

    # 1. 转换网格数据
    grid = convert_grid_to_geojson()

    if grid is not None:
        # 2. 生成网格元数据
        generate_grid_metadata(grid)

        # 3. 加载或生成预测数据
        predictions = load_real_predictions()
        if predictions is None:
            predictions = generate_sample_predictions(grid, days=7)

        print("\n" + "="*60)
        print("Data preparation complete!")
        print("="*60)
        print(f"Grid cells: {len(grid)}")
        print(f"Prediction days: {len(predictions)}")
        print("\nNext step: Run 'python app.py' to start the web server")
