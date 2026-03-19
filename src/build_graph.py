import geopandas as gpd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity


# 读取grid数据
grid = gpd.read_file("data/processed/chicago_grid.shp")
N = len(grid)
print("grid number:",N)

# 构建空间邻接图
def build_adj_adaptive():
    A_spatial = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if grid.geometry[i].touches(grid.geometry[j]):
                A_spatial[i,j] = 1
    
    poi = np.load("data/processed/poi_features.npy")
    landuse = np.load("data/processed/landuse_features.npy")
    night = np.load("data/processed/nightlight_features.npy")
    road = np.load("data/processed/road_density.npy")
    green = np.load("data/processed/green_ratio.npy")
    green_squared = green ** 2  # 二次项 体现双向作用
    feature = np.hstack([
        poi,
        landuse,
        night,
        road,
        green,
        green_squared
    ])
    # 计算特征相似度
    S_feature = cosine_similarity(feature)
    A_adaptive = A_spatial * S_feature
    A_adaptive = A_adaptive.astype(np.float32)
    np.save("data/processed/adj_adaptive.npy",A_adaptive)

# 距离衰减图
def build_adj_distance():
    coords = np.array([
        [geom.centroid.x, geom.centroid.y]
        for geom in grid.geometry
    ])
    dist = euclidean_distances(coords)
    sigma = dist.mean()
    A_distance = np.exp(-(dist**2)/(sigma**2))
    A_distance = A_distance.astype(np.float32)
    np.save("data/processed/adj_distance.npy",A_distance)

# 犯罪传播图
# A crime diffusion graph was constructed by measuring the Pearson correlation between crime time series across spatial units.
# def build_adj_crime():
#     crime = np.load("data/processed/crime_grid_timeseries.npy")
#     corr = np.corrcoef(crime.T)
#     A_crime = np.where(corr > 0.3, corr, 0)
#     np.save("data/processed/adj_crime.npy",A_crime)

# 改进为动态犯罪传播图，使用滚动窗口计算时间相关性，捕捉犯罪模式的动态变化
# def dynamic_adj_crime_gaussian(window=30, sigma=None):
#     """
#     构建犯罪动态图，使用 Gaussian similarity 替代相关系数
#     Args:
#         window: 时间窗口大小
#         sigma: 高斯核带宽，若为 None，则使用窗口内所有元素的 std 作为 sigma
#     Returns:
#         A_crime_dynamic: (T-window, N, N)
#     """
#     crime = np.load("data/processed/crime_grid_timeseries.npy")  # (T, N)
#     T, N = crime.shape
#     A_crime_list = []

#     for t in range(window, T):
#         crime_window = crime[t-window:t]       # (window, N)
#         crime_window = crime_window.T           # (N, window)

#         # 计算 sigma
#         if sigma is None:
#             sigma_val = np.std(crime_window)
#             if sigma_val == 0:
#                 sigma_val = 1.0
#         else:
#             sigma_val = sigma

#         # pairwise squared distance
#         diff = crime_window[:, np.newaxis, :] - crime_window[np.newaxis, :, :]  # (N,N,window)
#         dist2 = np.sum(diff**2, axis=2)  # (N,N)

#         # Gaussian similarity
#         sim = np.exp(-dist2 / (2*sigma_val**2)).astype(np.float32)

#         # 负值置0（理论上不可能有负值，但保险起见）
#         sim[sim < 0] = 0.0

#         A_crime_list.append(sim)

#     A_crime_dynamic = np.array(A_crime_list, dtype=np.float32)  # (T-window, N, N)
#     np.save("data/processed/adj_crime_dynamic_gaussian.npy", A_crime_dynamic)
#     print("Gaussian crime dynamic graph saved, shape:", A_crime_dynamic.shape)
#     return A_crime_dynamic

# 改进为融合OD流的动态犯罪传播图，使用 OD 流量相似度加权犯罪相关性，捕捉空间单元之间的动态交互影响
import numpy as np

def dynamic_adj_crime_gaussian_od(
    crime_path,
    bike_in_path,
    bike_out_path,
    taxi_in_path,
    taxi_out_path,
    window=30,
    sigma=None,
    daily_steps=24,
    K=10,
    alpha=0.5
):
    """
    构建动态犯罪图，融合 OD Top-K 特征
    Args:
        crime_path: 网格犯罪时间序列 npy (高频/日级都可)
        bike_in_path/out_path: 单车流量 npy
        taxi_in_path/out_path: 出租车流量 npy
        window: 时间窗口大小
        sigma: 高斯核带宽 (None 则自动用窗口 std)
        daily_steps: 高频流量 -> 日聚合步数
        K: Top-K 每行保留边数
        alpha: Gaussian犯罪图与OD图混合权重 (0~1)
    Returns:
        A_dynamic: (T-window, N, N) 动态犯罪图
    """
    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    crime = np.load(crime_path)  # (T, N)
    bike_in = np.load(bike_in_path)
    bike_out = np.load(bike_out_path)
    taxi_in = np.load(taxi_in_path)
    taxi_out = np.load(taxi_out_path)

    # 对齐最小长度并按日聚合
    min_len = min(len(bike_in), len(bike_out), len(taxi_in), len(taxi_out))
    print("=="*10,len(bike_in))
    crime = crime[:min_len]
    bike_in = bike_in[:min_len]
    bike_out = bike_out[:min_len]
    taxi_in = taxi_in[:min_len]
    taxi_out = taxi_out[:min_len]

    num_days = min_len // daily_steps
    N = crime.shape[1]

    def daily_sum(arr):
        arr = arr[:num_days*daily_steps]
        return arr.reshape(num_days, daily_steps, N).sum(axis=1)

    bike_in_daily = daily_sum(bike_in)
    bike_out_daily = daily_sum(bike_out)
    taxi_in_daily = daily_sum(taxi_in)
    taxi_out_daily = daily_sum(taxi_out)
    crime_daily = crime  # 如果crime已经日级，可直接用

    # -----------------------------
    # 2. 构建每日 OD 矩阵
    # -----------------------------
    OD_matrix = np.zeros((num_days, N, N), dtype=np.float32)
    for t in range(num_days):
        features = np.stack([
            bike_in_daily[t],
            bike_out_daily[t],
            taxi_in_daily[t],
            taxi_out_daily[t]
        ], axis=1)  # (N,4)
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (norm + 1e-6)
        OD_matrix[t] = features_norm @ features_norm.T

    # Top-K + 行归一化
    OD_topk = np.zeros_like(OD_matrix)
    for t in range(num_days):
        for i in range(N):
            topk_idx = np.argsort(OD_matrix[t,i])[-K:]
            OD_topk[t,i,topk_idx] = OD_matrix[t,i,topk_idx]
        OD_topk[t] = OD_topk[t] / (OD_topk[t].sum(axis=1, keepdims=True) + 1e-6)

    # -----------------------------
    # 3. 构建 Gaussian 犯罪动态图
    # -----------------------------
    A_crime_list = []
    for t in range(window, num_days):
        crime_window = crime_daily[t-window:t].T  # (N, window)
        sigma_val = np.std(crime_window) if sigma is None else sigma
        if sigma_val == 0:
            sigma_val = 1.0
        diff = crime_window[:, np.newaxis, :] - crime_window[np.newaxis, :, :]  # (N,N,window)
        dist2 = np.sum(diff**2, axis=2)
        sim = np.exp(-dist2/(2*sigma_val**2)).astype(np.float32)
        sim[sim<0] = 0.0
        A_crime_list.append(sim)
    A_crime = np.array(A_crime_list, dtype=np.float32)  # (T-window, N, N)

    # -----------------------------
    # 4. 混合 Gaussian + OD Top-K
    # -----------------------------
    # OD_topk 对齐时间 (从 window 开始)
    OD_topk_trim = OD_topk[window:]
    A_dynamic = alpha * A_crime + (1-alpha) * OD_topk_trim
    # 行归一化
    for t in range(A_dynamic.shape[0]):
        A_dynamic[t] = A_dynamic[t] / (A_dynamic[t].sum(axis=1, keepdims=True)+1e-6)

    # -----------------------------
    # 5. 保存
    # -----------------------------
    np.save("data/processed/adj_crime_dynamic_od.npy", A_dynamic)
    print("Dynamic crime+OD graph saved, shape:", A_dynamic.shape)
    return A_dynamic

# def build_all_adj():
    # build_adj_adaptive()
    # build_adj_distance()
    # build_adj_crime()
    # dynamic_adj_crime()
    # A_spatial = np.load("data/processed/adj_adaptive.npy")
    # A_distance = np.load("data/processed/adj_distance.npy")
    # A_crime = np.load("data/processed/adj_crime_dynamic.npy")
    # A = 0.5*A_spatial + 0.3*A_distance + 0.2*A_crime 
    # np.save("data/processed/mixed_graph_adj_matrix.npy", A.astype(np.float32))
# dynamic_adj_crime_gaussian()

dynamic_adj_crime_gaussian_od(
    crime_path="data/processed/crime_grid_timeseries.npy",
    bike_in_path="data/processed/bike_inflow_1246.npy",
    bike_out_path="data/processed/bike_outflow_1246.npy",
    taxi_in_path="data/processed/taxi_inflow_1246.npy",
    taxi_out_path="data/processed/taxi_outflow_1246.npy",
    window=30,
    sigma=None,
    daily_steps=24,
    K=10,
    alpha=0.5
)

# if __name__ == "__main__":
#     build_all_adj()

# 因为图注意力融合是在模型结构中融合，所以这里不再预先融合邻接矩阵，而是直接在模型中加载三个邻接矩阵进行融合