import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# -------------------------
# 读取空间特征
# -------------------------
poi = np.load("data/processed/poi_features.npy")
road = np.load("data/processed/road_density.npy")
landuse = np.load("data/processed/landuse_features.npy")
green = np.load("data/processed/green_features.npy")
nightlight = np.load("data/processed/nightlight_features.npy")
cameras = np.load("data/processed/camera_features.npy")

# -------------------------
# NaN检测与处理
# -------------------------
# 在 check_and_fix_nan 函数中针对绿地特殊处理
def check_and_fix_nan(features, name):
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        if name == "green":
            # 绿地NaN更可能是"无绿地"而非数据缺失
            print(f"警告: {name} 包含 {nan_count} 个NaN，使用0填充（表示无绿地）")
            features = np.nan_to_num(features, nan=0.0)
        else:
            print(f"警告: {name} 包含 {nan_count} 个NaN，使用均值填充")
            imputer = SimpleImputer(strategy='mean')
            features = imputer.fit_transform(features) if features.ndim > 1 else imputer.fit_transform(features.reshape(-1, 1)).flatten()
    return features


# 对每个特征进行NaN检查和处理
poi = check_and_fix_nan(poi, "poi")
road = check_and_fix_nan(road, "road")
landuse = check_and_fix_nan(landuse, "landuse")
green = check_and_fix_nan(green, "green")
nightlight = check_and_fix_nan(nightlight, "nightlight")
cameras = check_and_fix_nan(cameras, "cameras")

print(f"poi shape: {poi.shape}, range: [{poi.min():.2f}, {poi.max():.2f}]")
print(f"cameras shape: {cameras.shape}, range: [{cameras.min():.2f}, {cameras.max():.2f}]")

# -------------------------
# 拼接空间特征
# -------------------------
spatial_features = np.hstack([
    poi,
    road,
    landuse,
    green,
    nightlight,
    cameras
])

# 再次检查是否有NaN（防止拼接后出现问题）
spatial_features = check_and_fix_nan(spatial_features, "spatial_features")

scaler = StandardScaler()
spatial_features = scaler.fit_transform(spatial_features) # 标准化空间特征
print("spatial_features:", spatial_features.shape)

# (N,F)
# -------------------------
# 读取犯罪时间序列-- 目前使用合并的版本，三维（T,N,2）
# -------------------------
crime = np.load("data/processed/crime_combined_timeseries.npy")
# (T,N)
T,N,channel = crime.shape

# -------------------------
# 构造滞后特征
# -------------------------
crime_lag = []
T_lag = 7
for i in range(T_lag):
    crime_lag.append(
        crime[i:T-T_lag+i]  # (T-T_lag, N, 2)
    )
crime_lag = np.stack(crime_lag, axis=-1)  # (T-T_lag, N, 2, 7)

# 将滞后维度展平: (T-T_lag, N, 2, 7) -> (T-T_lag, N, 14)
# 这样暴力犯罪和财产犯罪的7天滞后都作为独立特征
T_final, N, n_crime_types, n_lag = crime_lag.shape
crime_lag = crime_lag.reshape(T_final, N, n_crime_types * n_lag)
print(f"crime_lag shape after reshape: {crime_lag.shape}")  # (T-T_lag, N, 14)
# -------------------------
# 读取气象数据
# -------------------------
weather = np.load("data/processed/weather_features.npy")
# (T,T_lag)
# rain, temperature, snow
weather = weather[T_lag:]
# -------------------------
# 扩展气象到空间维
# -------------------------
weather_expand = np.repeat(
    weather[:,np.newaxis,:],
    N,
    axis=1
)

# -------------------------
# 扩展空间特征到时间维
# -------------------------
spatial_expand = np.repeat(
    spatial_features[np.newaxis,:,:],
    T-T_lag,
    axis=0
)
print(spatial_expand.shape)  # (T-T_lag,N,F)
# (T-T_lag,N,F)

# 拼接OD流特征
# 对齐时间步（滞后窗口）
# 以滞后7天为例

# dynamic_od_lag = []
# dynamic_od_daily = np.load("data/processed/dynamic_od_flow_1246.npy")
# T_day = dynamic_od_daily.shape[0]
# for i in range(T_lag):
#     dynamic_od_lag.append(dynamic_od_daily[i:T_day-T_lag+i])
# dynamic_od_lag = np.stack(dynamic_od_lag, axis=-1)  # (T-T_lag, N, 4, 7)

# # 如果希望将 4 个通道和 7 天滞后展开成单一维度,4D->2D
# T_final, N, C, L = dynamic_od_lag.shape
# dynamic_od_lag_reshape = dynamic_od_lag.reshape(T_final, N, C*L)  # (T-T_lag, N, 12)

# print("="*10,dynamic_od_lag_reshape.shape)

# -------------------------
# 拼接最终输入
# -------------------------
X = np.concatenate([
    spatial_expand,
    weather_expand,
    crime_lag
],axis=2)

# label
Y = crime[T_lag:]

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generate_hypergraph_matrix(spatial_features, n_clusters=10):
    """
    基于空间特征聚类生成超图的 clique expansion 矩阵
    spatial_features: (N, F_static)
    """
    # 1. 检查并修复NaN
    if np.isnan(spatial_features).any():
        print(f"警告: generate_hypergraph_matrix 检测到NaN，使用均值填充")
        imputer = SimpleImputer(strategy='mean')
        spatial_features = imputer.fit_transform(spatial_features)

    # 2. 标准化特征
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(spatial_features)

    # 3. 确保标准化后没有NaN（防止所有值相同导致NaN）
    if np.isnan(feat_norm).any():
        print(f"警告: StandardScaler产生NaN，使用0填充")
        feat_norm = np.nan_to_num(feat_norm, nan=0.0)

    # 4. K-Means 聚类：将环境相似的网格聚在一起
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feat_norm)
    
    # 3. 构造关联矩阵 H (N x n_clusters)
    N = spatial_features.shape[0]
    H = np.zeros((N, n_clusters))
    for i in range(N):
        H[i, clusters[i]] = 1
        
    # 4. 构造 Clique Expansion 邻接矩阵 A_hg = H * H^T
    # 这种方式将属于同一个超边的点两两相连
    A_hg = np.dot(H, H.T)
    
    # 5. 归一化 (保持数值稳定)
    D = np.array(A_hg.sum(1))
    D_inv = np.power(D, -0.5).flatten()
    D_inv[np.isinf(D_inv)] = 0.
    D_mat_inv = np.diag(D_inv)
    A_hg_norm = D_mat_inv.dot(A_hg).dot(D_mat_inv)
    
    return A_hg_norm

A_hg = generate_hypergraph_matrix(spatial_features, n_clusters=12) # 12个功能区超边
print("A_hg shape:", A_hg.shape)
np.save("data/processed/adj_hypergraph.npy", A_hg)

static_idx = spatial_features.shape[1] 
print(f"Static features boundary (static_idx): {static_idx}")

print("X shape:",X.shape)
print("Y shape:",Y.shape)

np.save("data/processed/X.npy",X)
np.save("data/processed/Y.npy",Y)