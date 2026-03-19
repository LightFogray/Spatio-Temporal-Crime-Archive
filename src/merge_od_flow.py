import numpy as np
def build_daily_od_topk(
    bike_in_path,
    bike_out_path,
    taxi_in_path,
    taxi_out_path,
    daily_steps=24,
    K=10
):
    """
    将高频出租车/共享单车流量聚合为日级 OD Top-K 矩阵
    Args:
        bike_in_path/out_path: 单车流入/流出 npy 文件路径 (高频, shape=(T_high,N))
        taxi_in_path/out_path: 出租车流入/流出 npy 文件路径
        daily_steps: 每日时间步数量（如 24 小时）
        K: Top-K 每行保留的边
    Returns:
        OD_topk: (T_days, N, N)
    """
    # 1. 加载数据
    bike_in = np.load(bike_in_path)
    bike_out = np.load(bike_out_path)
    taxi_in = np.load(taxi_in_path)
    taxi_out = np.load(taxi_out_path)

    # 2. 对齐最小长度并按日聚合
    min_len = min(len(bike_in), len(bike_out), len(taxi_in), len(taxi_out))
    bike_in = bike_in[:min_len]
    bike_out = bike_out[:min_len]
    taxi_in = taxi_in[:min_len]
    taxi_out = taxi_out[:min_len]

    # reshape -> 日聚合
    num_days = min_len // daily_steps
    N = bike_in.shape[1]

    def daily_sum(arr):
        arr = arr[:num_days*daily_steps]  # 截断
        return arr.reshape(num_days, daily_steps, N).sum(axis=1)

    bike_in_daily = daily_sum(bike_in)
    bike_out_daily = daily_sum(bike_out)
    taxi_in_daily = daily_sum(taxi_in)
    taxi_out_daily = daily_sum(taxi_out)

    # 3. 构建每日 OD 矩阵 (节点间特征相似度)
    OD_matrix = np.zeros((num_days, N, N), dtype=np.float32)

    for t in range(num_days):
        # 每个节点特征: [bike_in, bike_out, taxi_in, taxi_out]
        features = np.stack([
            bike_in_daily[t],
            bike_out_daily[t],
            taxi_in_daily[t],
            taxi_out_daily[t]
        ], axis=1)  # (N,4)

        # 归一化
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (norm + 1e-6)

        # 节点间相似度
        OD_matrix[t] = features_norm @ features_norm.T

    # 4. Top-K 筛选 + 行归一化
    OD_topk = np.zeros_like(OD_matrix)
    for t in range(num_days):
        for i in range(N):
            topk_idx = np.argsort(OD_matrix[t, i])[-K:]
            OD_topk[t, i, topk_idx] = OD_matrix[t, i, topk_idx]
        # 行归一化
        OD_topk[t] = OD_topk[t] / (OD_topk[t].sum(axis=1, keepdims=True) + 1e-6)

    # 5. 保存
    np.save("data/processed/OD_topk_daily.npy", OD_topk)
    print(f"OD Top-K daily saved, shape: {OD_topk.shape}")

    return OD_topk


build_daily_od_topk(
    bike_in_path="data/processed/bike_inflow_1246.npy",
    bike_out_path="data/processed/bike_outflow_1246.npy",
    taxi_in_path="data/processed/taxi_inflow_1246.npy",
    taxi_out_path="data/processed/taxi_outflow_1246.npy",
    daily_steps=24,
    K=10
)

# =========================
# 1. 加载已对齐的 OD 流量 npy
# =========================
# bike_in = np.load("data/processed/bike_inflow_1246.npy")      # (T_hour, N)
# bike_out = np.load("data/processed/bike_outflow_1246.npy")   # (T_hour, N)
# taxi_in = np.load("data/processed/taxi_inflow_1246.npy")     # (T_hour, N)
# taxi_out = np.load("data/processed/taxi_outflow_1246.npy")   # (T_hour, N)

# N = bike_in.shape[1]
# T_hour = bike_in.shape[0]

# # =========================
# # 2. 聚合到天
# # =========================
# hours_per_day = 24
# T_day = T_hour // hours_per_day  # 向下取整，多余小时舍弃

# def hourly_to_daily(flow_hour):
#     T_hour, N = flow_hour.shape
#     hours_per_day = 24
#     remainder = T_hour % hours_per_day
#     if remainder != 0:
#         # 用 0 补齐最后一天
#         padding = np.zeros((hours_per_day - remainder, N), dtype=flow_hour.dtype)
#         flow_hour = np.vstack([flow_hour, padding])
#     T_day = flow_hour.shape[0] // hours_per_day
#     flow_daily = flow_hour.reshape(T_day, hours_per_day, N).sum(axis=1)
#     return flow_daily

# bike_in_day = hourly_to_daily(bike_in)
# bike_out_day = hourly_to_daily(bike_out)
# taxi_in_day = hourly_to_daily(taxi_in)
# taxi_out_day = hourly_to_daily(taxi_out)

# print("Daily shapes:", bike_in_day.shape, bike_out_day.shape, taxi_in_day.shape, taxi_out_day.shape)

# # =========================
# # 3. 合并为 dynamic features
# # =========================
# # F_dynamic = 4: [bike_in, bike_out, taxi_in, taxi_out]
# dynamic_features = np.stack([bike_in_day, bike_out_day, taxi_in_day, taxi_out_day], axis=-1)
# print("Dynamic features shape:", dynamic_features.shape)  # (T_day, N, 4)

# # =========================
# # 4. 去极值 + 标准化 (可选)
# # =========================
# def normalize_feature(x):
#     # log1p + clip + z-score
#     x_log = np.log1p(x)
#     x_clip = np.clip(x_log, 0, np.percentile(x_log, 99))
#     mean = x_clip.mean(axis=0, keepdims=True)
#     std = x_clip.std(axis=0, keepdims=True) + 1e-6
#     return (x_clip - mean) / std

# dynamic_features_norm = normalize_feature(dynamic_features)
# print("Normalized dynamic features shape:", dynamic_features_norm.shape)

# # =========================
# # 5. 保存 npy
# # =========================
# np.save("data/processed/dynamic_od_flow_1246.npy", dynamic_features_norm)
# print("✅ Dynamic features saved as dynamic_flow_1246.npy")