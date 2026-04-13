#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OD 数据网格映射诊断脚本
========================
"""

import pandas as pd
import numpy as np

# 配置
GRID_SIZE = 0.01
LON_MIN, LAT_MIN = -88, 41
NUM_X, NUM_Y = 100, 100
N = NUM_X * NUM_Y

print("="*60)
print("OD 数据网格映射诊断")
print("="*60)
print(f"网格参数: {NUM_X}x{NUM_Y} = {N} 个网格")
print(f"  Lon范围: [{LON_MIN}, {LON_MIN + NUM_X*GRID_SIZE}]")
print(f"  Lat范围: [{LAT_MIN}, {LAT_MIN + NUM_Y*GRID_SIZE}]")

# 检查 bike 数据
print("\n" + "="*60)
print("检查 Bike 数据")
print("="*60)

try:
    df_bike = pd.read_csv("./OD_data/divvy_clean_2022_2023.csv", nrows=1000)
    print(f"\n列名: {list(df_bike.columns)}")
    print(f"\n前5行:")
    print(df_bike.head())

    # 检查坐标范围
    print(f"\n坐标统计:")
    print(f"  start_lat: [{df_bike['start_lat'].min():.4f}, {df_bike['start_lat'].max():.4f}]")
    print(f"  start_lng: [{df_bike['start_lng'].min():.4f}, {df_bike['start_lng'].max():.4f}]")
    print(f"  end_lat:   [{df_bike['end_lat'].min():.4f}, {df_bike['end_lat'].max():.4f}]")
    print(f"  end_lng:   [{df_bike['end_lng'].min():.4f}, {df_bike['end_lng'].max():.4f}]")

    # 测试网格映射
    print(f"\n网格映射测试 (前5条):")
    for i in range(min(5, len(df_bike))):
        lon = df_bike['start_lng'].iloc[i]
        lat = df_bike['start_lat'].iloc[i]
        x = int((lon - LON_MIN) / GRID_SIZE)
        y = int((lat - LAT_MIN) / GRID_SIZE)
        grid_id = x + y * NUM_X
        print(f"  ({lon:.4f}, {lat:.4f}) -> x={x}, y={y}, grid_id={grid_id}")
        in_bounds = (0 <= x < NUM_X) and (0 <= y < NUM_Y)
        print(f"    在网格范围内: {in_bounds}")

except Exception as e:
    print(f"错误: {e}")

# 检查 taxi 数据
print("\n" + "="*60)
print("检查 Taxi 数据")
print("="*60)

try:
    df_taxi = pd.read_csv("./OD_data/taxi_clean_2022_2023.csv", nrows=1000)
    print(f"\n列名: {list(df_taxi.columns)}")
    print(f"\n前5行:")
    print(df_taxi.head())

    # 检查坐标范围
    print(f"\n坐标统计:")
    print(f"  Pickup Lat:  [{df_taxi['Pickup Centroid Latitude'].min():.4f}, {df_taxi['Pickup Centroid Latitude'].max():.4f}]")
    print(f"  Pickup Lon:  [{df_taxi['Pickup Centroid Longitude'].min():.4f}, {df_taxi['Pickup Centroid Longitude'].max():.4f}]")
    print(f"  Dropoff Lat: [{df_taxi['Dropoff Centroid Latitude'].min():.4f}, {df_taxi['Dropoff Centroid Latitude'].max():.4f}]")
    print(f"  Dropoff Lon: [{df_taxi['Dropoff Centroid Longitude'].min():.4f}, {df_taxi['Dropoff Centroid Longitude'].max():.4f}]")

    # 测试网格映射
    print(f"\n网格映射测试 (前5条):")
    for i in range(min(5, len(df_taxi))):
        lon = df_taxi['Pickup Centroid Longitude'].iloc[i]
        lat = df_taxi['Pickup Centroid Latitude'].iloc[i]
        x = int((lon - LON_MIN) / GRID_SIZE)
        y = int((lat - LAT_MIN) / GRID_SIZE)
        grid_id = x + y * NUM_X
        print(f"  ({lon:.4f}, {lat:.4f}) -> x={x}, y={y}, grid_id={grid_id}")
        in_bounds = (0 <= x < NUM_X) and (0 <= y < NUM_Y)
        print(f"    在网格范围内: {in_bounds}")

except Exception as e:
    print(f"错误: {e}")

# 检查 crime 网格
print("\n" + "="*60)
print("检查 Crime 网格系统")
print("="*60)

try:
    import geopandas as gpd
    grid = gpd.read_file("data/processed/chicago_grid.shp")
    print(f"\nCrime 网格数量: {len(grid)}")
    print(f"网格列: {list(grid.columns)}")

    # 获取网格边界
    bounds = grid.total_bounds
    print(f"\n网格边界: {bounds}")  # [minx, miny, maxx, maxy]
    print(f"  Lon范围: [{bounds[0]:.4f}, {bounds[2]:.4f}]")
    print(f"  Lat范围: [{bounds[1]:.4f}, {bounds[3]:.4f}]")

    # 计算网格分辨率
    sample_grid = grid.iloc[0]
    geom = sample_grid.geometry
    print(f"\n示例网格几何: {geom}")

except Exception as e:
    print(f"错误: {e}")
    print("提示: 需要安装 geopandas: pip install geopandas")

print("\n" + "="*60)
print("诊断完成")
print("="*60)
