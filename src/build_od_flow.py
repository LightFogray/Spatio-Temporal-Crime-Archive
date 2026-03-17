import pandas as pd
import numpy as np
from collections import defaultdict

# ==============================
# 1️⃣ 参数设置（根据你数据改）
# ==============================
GRID_SIZE = 0.01
LON_MIN, LAT_MIN = -88, 41   # Chicago 大致范围
NUM_X = 200                  # 横向网格数（根据范围调整）
N = 1246                     # 你的节点数

# ==============================
# 2️⃣ 坐标 → grid id
# ==============================
def coord_to_grid(lon, lat):
    try:
        x = int((lon - LON_MIN) / GRID_SIZE)
        y = int((lat - LAT_MIN) / GRID_SIZE)
        gid = y * NUM_X + x

        if gid < 0 or gid >= N:
            return None
        return gid
    except:
        return None

# ==============================
# 3️⃣ 时间离散
# ==============================
def time_to_bin(ts):
    try:
        return pd.to_datetime(ts).floor("H")
    except:
        return None

# ==============================
# 4️⃣ 构建 OD
# ==============================
def build_od(df, start_lon, start_lat, end_lon, end_lat, time_col):
    OD = defaultdict(lambda: np.zeros((N, N)))

    for _, row in df.iterrows():
        i = coord_to_grid(row[start_lon], row[start_lat])
        j = coord_to_grid(row[end_lon], row[end_lat])
        t = time_to_bin(row[time_col])

        if i is None or j is None or t is None:
            continue

        OD[t][i, j] += 1

    return OD

# ==============================
# 5️⃣ OD → flow
# ==============================
def od_to_flow_fixed(OD, full_time):
    T = len(full_time)

    inflow = np.zeros((T, N))
    outflow = np.zeros((T, N))

    for t_idx, t in enumerate(full_time):
        if t in OD:
            od = OD[t]
            inflow[t_idx] = od.sum(axis=0)
            outflow[t_idx] = od.sum(axis=1)

    return inflow, outflow

# ==============================
# 6️⃣ 共享单车
# ==============================
print("Loading bike data...")
bike_df = pd.read_csv("bike.csv")

bike_od = build_od(
    bike_df,
    "start_lng", "start_lat",
    "end_lng", "end_lat",
    "started_at"
)



# ==============================
# 7️⃣ 出租车
# ==============================
print("Loading taxi data...")
taxi_df = pd.read_csv("taxi.csv")

# 清洗
taxi_df = taxi_df.dropna(subset=[
    "Pickup Centroid Latitude",
    "Pickup Centroid Longitude",
    "Dropoff Centroid Latitude",
    "Dropoff Centroid Longitude"
])

taxi_od = build_od(
    taxi_df,
    "Pickup Centroid Longitude",
    "Pickup Centroid Latitude",
    "Dropoff Centroid Longitude",
    "Dropoff Centroid Latitude",
    "Trip Start Timestamp"
)

full_time = sorted(list(set(bike_od.keys()) | set(taxi_od.keys())))

bike_in, bike_out = od_to_flow_fixed(bike_od, full_time)
taxi_in, taxi_out = od_to_flow_fixed(taxi_od, full_time)
# ===== 去极值（log）=====
bike_in = np.log1p(bike_in)
bike_out = np.log1p(bike_out)
taxi_in = np.log1p(taxi_in)
taxi_out = np.log1p(taxi_out)

# ===== 标准化 =====
def normalize(x):
    mean = x.mean()
    std = x.std() + 1e-6
    return (x - mean) / std

bike_in = normalize(bike_in)
bike_out = normalize(bike_out)
taxi_in = normalize(taxi_in)
taxi_out = normalize(taxi_out)

np.save("bike_inflow.npy", bike_in)
np.save("bike_outflow.npy", bike_out)
print("Bike done:", bike_in.shape)

np.save("taxi_inflow.npy", taxi_in)
np.save("taxi_outflow.npy", taxi_out)
print("Taxi done:", taxi_in.shape)