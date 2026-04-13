#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OD数据日期过滤脚本
==================

功能：
1. 读取 divvy_clean.csv 和 taxi_clean.csv
2. 过滤 2022-01-01 至 2023-12-31 的数据
3. 保存过滤后的文件

输入文件格式：
- divvy_clean.csv: started_at,start_lat,start_lng,end_lat,end_lng
- taxi_clean.csv: Trip Start Timestamp,Pickup Centroid Latitude,Pickup Centroid Longitude,Dropoff Centroid Latitude,Dropoff Centroid Longitude

作者：犯罪地理学数据分析系统
日期：2026-04-12
"""

import pandas as pd
import os
from datetime import datetime

# ==============================
# 配置参数
# ==============================

# 输入文件路径
INPUT_FILES = {
    "bike": "../OD_data/divvy_clean.csv",
    "taxi": "../OD_data/taxi_clean.csv"
}

# 输出文件路径
OUTPUT_FILES = {
    "bike": "../OD_data/divvy_clean_2022_2023.csv",
    "taxi": "../OD_data/taxi_clean_2022_2023.csv"
}

# 日期范围
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# 日期列名
DATE_COLUMNS = {
    "bike": "started_at",
    "taxi": "Trip Start Timestamp"
}

# 日期格式
DATE_FORMATS = {
    "bike": "%Y-%m-%d %H:%M:%S",  # 2021-01-23 16:14:19
    "taxi": "%m/%d/%Y %I:%M:%S %p"  # 01/01/2022 12:00:00 AM
}

# ==============================
# 过滤函数
# ==============================

def filter_by_date(input_path, output_path, date_column, date_format, chunk_size=500000):
    """
    按日期范围过滤CSV文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        date_column: 日期列名
        date_format: 日期格式字符串
        chunk_size: 分块大小

    Returns:
        过滤后的DataFrame
    """
    print(f"\n处理文件: {input_path}")

    if not os.path.exists(input_path):
        print(f"  错误: 文件不存在 {input_path}")
        return None

    # 解析日期范围
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    print(f"  日期范围: {START_DATE} 至 {END_DATE}")

    filtered_chunks = []
    total_rows = 0
    filtered_rows = 0

    # 分块读取处理
    chunks = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(chunks):
        print(f"  处理块 {i+1}...")
        total_rows += len(chunk)

        # 检查日期列是否存在
        if date_column not in chunk.columns:
            print(f"  错误: 列 '{date_column}' 不存在")
            print(f"  可用列: {list(chunk.columns)}")
            return None

        # 转换日期
        try:
            chunk[date_column] = pd.to_datetime(
                chunk[date_column],
                format=date_format,
                errors='coerce'
            )
        except Exception as e:
            # 如果指定格式失败，尝试自动推断
            print(f"  警告: 使用指定格式失败，尝试自动推断: {e}")
            chunk[date_column] = pd.to_datetime(chunk[date_column], errors='coerce')

        # 过滤日期范围
        mask = (chunk[date_column] >= start_date) & (chunk[date_column] <= end_date)
        chunk_filtered = chunk[mask].copy()

        filtered_rows += len(chunk_filtered)
        filtered_chunks.append(chunk_filtered)

        print(f"    原始: {len(chunk):,} 条, 过滤后: {len(chunk_filtered):,} 条")

    # 合并所有块
    result_df = pd.concat(filtered_chunks, ignore_index=True)

    # 保存结果
    result_df.to_csv(output_path, index=False)

    print(f"  完成!")
    print(f"    总行数: {total_rows:,}")
    print(f"    保留行数: {filtered_rows:,}")
    print(f"    过滤比例: {filtered_rows/total_rows*100:.1f}%")
    print(f"    保存到: {output_path}")

    return result_df


def quick_filter(input_path, output_path, date_column, date_format):
    """
    快速过滤（适用于内存足够的情况）
    """
    print(f"\n处理文件: {input_path}")

    if not os.path.exists(input_path):
        print(f"  错误: 文件不存在 {input_path}")
        return None

    # 读取数据
    print(f"  读取数据...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  原始行数: {len(df):,}")

    # 转换日期
    print(f"  转换日期格式: {date_format}")
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
    except:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # 过滤
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    df_filtered = df[mask].copy()

    print(f"  过滤后行数: {len(df_filtered):,}")
    print(f"  保留比例: {len(df_filtered)/len(df)*100:.1f}%")

    # 保存
    df_filtered.to_csv(output_path, index=False)
    print(f"  保存到: {output_path}")

    return df_filtered


# ==============================
# 主流程
# ==============================

if __name__ == "__main__":
    print("="*60)
    print("OD数据日期过滤")
    print("="*60)
    print(f"目标日期范围: {START_DATE} 至 {END_DATE}")

    results = {}

    for key in INPUT_FILES:
        # 根据文件大小选择处理方式
        file_size = os.path.getsize(INPUT_FILES[key]) if os.path.exists(INPUT_FILES[key]) else 0
        file_size_mb = file_size / (1024 * 1024)

        print(f"\n{'='*60}")
        print(f"处理 {key.upper()} 数据")
        print(f"文件大小: {file_size_mb:.1f} MB")

        if file_size_mb > 500:  # 大于500MB使用分块处理
            df = filter_by_date(
                INPUT_FILES[key],
                OUTPUT_FILES[key],
                DATE_COLUMNS[key],
                DATE_FORMATS[key]
            )
        else:
            df = quick_filter(
                INPUT_FILES[key],
                OUTPUT_FILES[key],
                DATE_COLUMNS[key],
                DATE_FORMATS[key]
            )

        results[key] = df

    # 汇总统计
    print(f"\n{'='*60}")
    print("处理完成汇总")
    print("="*60)
    for key, df in results.items():
        if df is not None:
            print(f"{key.upper()}:")
            print(f"  保留记录: {len(df):,} 条")
            print(f"  输出文件: {OUTPUT_FILES[key]}")
            print()
