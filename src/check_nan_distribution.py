#!/usr/bin/env python3
"""
检查所有特征文件中的 NaN 分布
==============================
"""

import numpy as np
import pandas as pd

def check_nans(filepath, name):
    """检查单个文件的NaN情况"""
    try:
        data = np.load(filepath)
        total_elements = data.size
        nan_count = np.isnan(data).sum()
        nan_ratio = nan_count / total_elements * 100

        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Total elements: {total_elements:,}")
        print(f"  NaN count: {nan_count:,}")
        print(f"  NaN ratio: {nan_ratio:.2f}%")

        # 如果是2D，检查每行/每列的NaN
        if len(data.shape) == 2:
            nan_per_row = np.isnan(data).sum(axis=1)
            nan_per_col = np.isnan(data).sum(axis=0)

            rows_with_nan = (nan_per_row > 0).sum()
            cols_with_nan = (nan_per_col > 0).sum()

            print(f"  Rows with NaN: {rows_with_nan}/{data.shape[0]} ({rows_with_nan/data.shape[0]*100:.1f}%)")
            print(f"  Cols with NaN: {cols_with_nan}/{data.shape[1]} ({cols_with_nan/data.shape[1]*100:.1f}%)")

            # 显示哪些列NaN最多
            if cols_with_nan > 0:
                worst_cols = np.argsort(nan_per_col)[-5:]
                print(f"  Worst columns (top 5): {worst_cols}, NaN counts: {nan_per_col[worst_cols]}")

        return nan_ratio

    except FileNotFoundError:
        print(f"\n{name}: FILE NOT FOUND - {filepath}")
        return None
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")
        return None


print("="*60)
print("检查所有特征文件中的 NaN 分布")
print("="*60)

files_to_check = [
    ("data/processed/poi_features.npy", "POI Features"),
    ("data/processed/landuse_features.npy", "Landuse Features"),
    ("data/processed/nightlight_features.npy", "Nightlight Features"),
    ("data/processed/road_density.npy", "Road Density"),
    ("data/processed/green_features.npy", "Green Features"),
    ("data/processed/green_ratio.npy", "Green Ratio"),
    ("data/processed/camera_features.npy", "Camera Features"),
    ("data/processed/dynamic_od_flow.npy", "OD Flow"),
    ("data/processed/crime_combined_timeseries.npy", "Crime Timeseries"),
    ("data/processed/weather_features.npy", "Weather Features"),
]

results = {}
for filepath, name in files_to_check:
    ratio = check_nans(filepath, name)
    if ratio is not None:
        results[name] = ratio

print("\n" + "="*60)
print("汇总评估")
print("="*60)

for name, ratio in sorted(results.items(), key=lambda x: x[1], reverse=True):
    status = "⚠️  HIGH" if ratio > 10 else ("⚡ MEDIUM" if ratio > 1 else "✅ OK")
    print(f"{status} {name:30s}: {ratio:6.2f}% NaN")

print("\n" + "="*60)
print("建议:")
print("  - NaN < 1%: 正常，用0填充即可")
print("  - 1% < NaN < 10%: 需要关注，建议检查数据来源")
print("  - NaN > 10%: 严重问题，需要排查数据处理流程")
print("="*60)
