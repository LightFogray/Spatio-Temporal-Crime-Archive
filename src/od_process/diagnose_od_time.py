#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OD数据时间完整性诊断脚本
==========================

诊断 bike/taxi 数据的时间缺失情况
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 配置
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

FILES = {
    "bike": "./OD_data/divvy_clean_2022_2023.csv",
    "taxi": "./OD_data/taxi_clean_2022_2023.csv"
}

DATE_COLS = {
    "bike": "started_at",
    "taxi": "Trip Start Timestamp"
}

DATE_FORMATS = {
    "bike": "%Y-%m-%d %H:%M:%S",
    "taxi": "%Y-%m-%d %H:%M:%S"
}

def diagnose_missing_hours(file_path, date_col, date_fmt, name):
    """诊断缺失的小时"""
    print(f"\n{'='*60}")
    print(f"诊断: {name}")
    print(f"{'='*60}")

    # 读取数据
    print(f"读取: {file_path}")
    df = pd.read_csv(file_path, usecols=[date_col])

    # 转换时间
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt, errors='coerce')
    df = df.dropna()

    #  floor 到小时
    df['hour'] = df[date_col].dt.floor('H')

    # 实际有的时间
    actual_hours = set(df['hour'].unique())
    print(f"实际记录的小时数: {len(actual_hours)}")

    # 期望的时间范围
    expected_range = pd.date_range(
        start=pd.to_datetime(START_DATE),
        end=pd.to_datetime(END_DATE) + timedelta(days=1),  # 包含最后一天
        freq='H'
    )
    expected_hours = set(expected_range)
    print(f"期望的小时数: {len(expected_hours)}")

    # 缺失的时间
    missing_hours = sorted(expected_hours - actual_hours)
    print(f"缺失的小时数: {len(missing_hours)}")

    if missing_hours:
        print(f"\n缺失时间段示例（前10个）:")
        for h in missing_hours[:10]:
            print(f"  {h}")

        # 按日期统计缺失
        missing_dates = [h.date() for h in missing_hours]
        date_counts = pd.Series(missing_dates).value_counts().sort_index()

        print(f"\n按日期统计缺失小时数:")
        print(f"  涉及 {len(date_counts)} 天")
        if len(date_counts) > 0:
            print(f"  缺失最多的日期: {date_counts.index[0]} ({date_counts.iloc[0]} 小时)")

        # 检查是否是特定时段缺失
        missing_hours_of_day = [h.hour for h in missing_hours]
        hour_dist = pd.Series(missing_hours_of_day).value_counts().sort_index()
        print(f"\n缺失的小时段分布:")
        for hour, count in hour_dist.head(5).items():
            print(f"  {hour:02d}:00 - {count} 次")

    return missing_hours

def generate_aligned_od_flow():
    """生成对齐的OD Flow特征"""
    print(f"\n{'='*60}")
    print("生成对齐的OD Flow特征")
    print(f"{'='*60}")

    # 加载已有的npy文件
    try:
        bike_in = np.load("data/processed/bike_inflow.npy")
        bike_out = np.load("data/processed/bike_outflow.npy")
        taxi_in = np.load("data/processed/taxi_inflow.npy")
        taxi_out = np.load("data/processed/taxi_outflow.npy")
    except FileNotFoundError as e:
        print(f"错误: 找不到npy文件: {e}")
        print("请先运行 build_od_flow.py")
        return

    print(f"原始形状:")
    print(f"  bike:  in={bike_in.shape}, out={bike_out.shape}")
    print(f"  taxi:  in={taxi_in.shape}, out={taxi_out.shape}")

    # 取最小长度对齐
    T_min = min(bike_in.shape[0], taxi_in.shape[0])
    print(f"\n对齐到最小时间步: {T_min}")

    bike_in = bike_in[:T_min]
    bike_out = bike_out[:T_min]
    taxi_in = taxi_in[:T_min]
    taxi_out = taxi_out[:T_min]

    # 合并为 (T, N, 4) 格式
    # 通道: 0=bike_in, 1=bike_out, 2=taxi_in, 3=taxi_out
    od_flow = np.stack([bike_in, bike_out, taxi_in, taxi_out], axis=-1)

    print(f"合并后形状: {od_flow.shape} (T={T_min}, N={od_flow.shape[1]}, channels=4)")

    # 保存
    output_path = "./OD_data/dynamic_od_flow.npy"
    np.save(output_path, od_flow.astype(np.float32))
    print(f"保存到: {output_path}")

    # 统计信息
    print(f"\nOD Flow统计:")
    print(f"  bike_in:  mean={bike_in.mean():.2f}, max={bike_in.max():.2f}")
    print(f"  bike_out: mean={bike_out.mean():.2f}, max={bike_out.max():.2f}")
    print(f"  taxi_in:  mean={taxi_in.mean():.2f}, max={taxi_in.max():.2f}")
    print(f"  taxi_out: mean={taxi_out.mean():.2f}, max={taxi_out.max():.2f}")

if __name__ == "__main__":
    print("="*60)
    print("OD数据时间完整性诊断")
    print("="*60)
    print(f"期望范围: {START_DATE} 至 {END_DATE}")

    # 诊断每个文件
    for name, filepath in FILES.items():
        if os.path.exists(filepath):
            diagnose_missing_hours(
                filepath,
                DATE_COLS[name],
                DATE_FORMATS[name],
                name
            )
        else:
            print(f"\n文件不存在: {filepath}")

    # 生成对齐的OD Flow
    import os
    generate_aligned_od_flow()

    print("\n" + "="*60)
    print("诊断完成!")
    print("="*60)
