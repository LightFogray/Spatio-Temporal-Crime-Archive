import numpy as np
import os

# =========================
# OD Flow 合并脚本
# =========================
# 输入: 日级的 OD flow (730 天, 已对齐到 1246 网格)
# 输出: 合并后的 dynamic_od_flow.npy (T, N, 4)

print("="*60)
print("OD Flow 合并（日级）")
print("="*60)

# 配置
data_dir = "data/processed"
bike_in_path = os.path.join(data_dir, "bike_inflow_1246.npy")
bike_out_path = os.path.join(data_dir, "bike_outflow_1246.npy")
taxi_in_path = os.path.join(data_dir, "taxi_inflow_1246.npy")
taxi_out_path = os.path.join(data_dir, "taxi_outflow_1246.npy")

# =========================
# 1. 加载 OD 流量 npy
# =========================
print("\n加载 OD flow 文件...")

try:
    bike_in = np.load(bike_in_path)
    bike_out = np.load(bike_out_path)
    taxi_in = np.load(taxi_in_path)
    taxi_out = np.load(taxi_out_path)
except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e}")
    print("请先运行 build_od_flow.py 生成 OD flow 文件")
    exit(1)

print(f"  bike_in:   {bike_in.shape}")
print(f"  bike_out:  {bike_out.shape}")
print(f"  taxi_in:   {taxi_in.shape}")
print(f"  taxi_out:  {taxi_out.shape}")

# =========================
# 2. 对齐时间步
# =========================
print("\n对齐时间步...")

T_bike_in = bike_in.shape[0]
T_bike_out = bike_out.shape[0]
T_taxi_in = taxi_in.shape[0]
T_taxi_out = taxi_out.shape[0]

T_min = min(T_bike_in, T_bike_out, T_taxi_in, T_taxi_out)

print(f"  bike_in:   {T_bike_in} 天")
print(f"  bike_out:  {T_bike_out} 天")
print(f"  taxi_in:   {T_taxi_in} 天")
print(f"  taxi_out:  {T_taxi_out} 天")
print(f"  取最小值:  {T_min} 天")

# 截取对齐
bike_in = bike_in[:T_min]
bike_out = bike_out[:T_min]
taxi_in = taxi_in[:T_min]
taxi_out = taxi_out[:T_min]

# =========================
# 3. 合并为 dynamic features (T, N, 4)
# =========================
print("\n合并特征...")

# 通道: 0=bike_in, 1=bike_out, 2=taxi_in, 3=taxi_out
dynamic_features = np.stack([bike_in, bike_out, taxi_in, taxi_out], axis=-1)

print(f"  合并后形状: {dynamic_features.shape}")
print(f"  时间跨度: {T_min} 天")

# =========================
# 4. 归一化 (log1p + z-score)
# =========================
print("\n归一化...")

def normalize_feature(x):
    """log1p + z-score 归一化"""
    # log1p 变换（处理长尾分布）
    x_log = np.log1p(x)

    # 可选：去除极端值
    x_clip = np.clip(x_log, 0, np.percentile(x_log, 99))

    # z-score 标准化 (对每个特征通道独立)
    mean = x_clip.mean(axis=(0, 1), keepdims=True)  # (1, 1, 4)
    std = x_clip.std(axis=(0, 1), keepdims=True) + 1e-6

    return (x_clip - mean) / std

dynamic_features_norm = normalize_feature(dynamic_features)

print(f"  归一化后范围: [{dynamic_features_norm.min():.2f}, {dynamic_features_norm.max():.2f}]")
print(f"  均值: {dynamic_features_norm.mean():.4f}, 标准差: {dynamic_features_norm.std():.4f}")

# =========================
# 5. 保存
# =========================
print("\n保存结果...")

output_path = os.path.join(data_dir, "dynamic_od_flow.npy")
np.save(output_path, dynamic_features_norm.astype(np.float32))

print(f"  保存到: {output_path}")
print(f"  最终形状: {dynamic_features_norm.shape} (T_day, N, 4)")

# 保存未归一化版本（可选）
output_raw_path = os.path.join(data_dir, "dynamic_od_flow_raw.npy")
np.save(output_raw_path, dynamic_features.astype(np.float32))
print(f"  原始值保存到: {output_raw_path}")

print("\n" + "="*60)
print("✅ OD Flow 合并完成！")
print("="*60)

# 统计信息
print("\n各通道统计:")
channel_names = ["bike_in", "bike_out", "taxi_in", "taxi_out"]
for i, name in enumerate(channel_names):
    ch = dynamic_features[:, :, i]
    print(f"  {name:12s}: mean={ch.mean():.2f}, max={ch.max():.2f}, non-zero={(ch>0).sum()/ch.size*100:.1f}%")
