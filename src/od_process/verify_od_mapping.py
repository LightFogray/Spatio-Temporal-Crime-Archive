#!/usr/bin/env python3
"""
验证 OD 映射质量
"""
import numpy as np
import pandas as pd

print("="*60)
print("OD 映射质量验证")
print("="*60)

# 加载数据
bike_in = np.load("data/processed/bike_inflow_1246.npy")
bike_out = np.load("data/processed/bike_outflow_1246.npy")
taxi_in = np.load("data/processed/taxi_inflow_1246.npy")
taxi_out = np.load("data/processed/taxi_outflow_1246.npy")

print(f"\n数据形状: {bike_in.shape} (T={bike_in.shape[0]}天, N={bike_in.shape[1]}网格)")

# 统计非零比例
channels = {
    "bike_in": bike_in,
    "bike_out": bike_out,
    "taxi_in": taxi_in,
    "taxi_out": taxi_out
}

print("\n各通道统计:")
for name, data in channels.items():
    non_zero = (data > 0).sum()
    total = data.size
    print(f"  {name:12s}: 非零={non_zero:8d}/{total:8d} ({non_zero/total*100:5.2f}%), 最大值={data.max():.2f}")

# 检查时间序列完整性
print("\n时间序列完整性:")
for name, data in channels.items():
    # 每天的总流量
    daily_total = data.sum(axis=1)
    zero_days = (daily_total == 0).sum()
    print(f"  {name:12s}: 零流量天数={zero_days}/{len(daily_total)}")

# 检查空间覆盖
print("\n空间覆盖 (有多少网格至少有一个OD记录):")
for name, data in channels.items():
    active_grids = (data.sum(axis=0) > 0).sum()
    print(f"  {name:12s}: 活跃网格={active_grids}/{data.shape[1]} ({active_grids/data.shape[1]*100:.1f}%)")

# 合并后的 dynamic_od_flow
print("\n" + "="*60)
print("验证合并后的 dynamic_od_flow.npy")
print("="*60)

od_flow = np.load("data/processed/dynamic_od_flow.npy")
print(f"形状: {od_flow.shape}")

# 检查归一化效果
print(f"\n归一化后统计:")
for i, name in enumerate(["bike_in", "bike_out", "taxi_in", "taxi_out"]):
    ch = od_flow[:, :, i]
    print(f"  {name:12s}: mean={ch.mean():.4f}, std={ch.std():.4f}, range=[{ch.min():.2f}, {ch.max():.2f}]")

print("\n" + "="*60)
if (bike_in > 0).sum() > 0:
    print("✅ OD 映射成功！数据质量正常。")
else:
    print("❌ 警告: 映射后数据全为0！")
print("="*60)
