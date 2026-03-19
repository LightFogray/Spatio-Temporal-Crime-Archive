import numpy as np
# -------------------------
# 读取空间特征
# -------------------------
poi = np.load("data/processed/poi_features.npy")
road = np.load("data/processed/road_density.npy")
landuse = np.load("data/processed/landuse_features.npy")
green = np.load("data/processed/green_ratio.npy")
nightlight = np.load("data/processed/nightlight_features.npy")
# -------------------------
# 拼接空间特征
# -------------------------
spatial_features = np.hstack([
    poi,
    road,
    landuse,
    green,
    nightlight
])
print("spatial_features:", spatial_features.shape)

# (N,F)
# -------------------------
# 读取犯罪时间序列
# -------------------------
crime = np.load("data/processed/crime_grid_timeseries.npy")
# (T,N)
T,N = crime.shape

# -------------------------
# 构造滞后特征
# -------------------------
crime_lag = []
T_lag = 7
for i in range(T_lag):

    crime_lag.append(
        crime[i:T-T_lag+i]
    )
crime_lag = np.stack(crime_lag,axis=-1)
# (T-T_lag,N,3)
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
# (T-T_lag,N,F)

# 拼接OD流特征
# 对齐时间步（滞后窗口）
# 以滞后7天为例

dynamic_od_lag = []
dynamic_od_daily = np.load("data/processed/dynamic_od_flow_1246.npy")
T_day = dynamic_od_daily.shape[0]
for i in range(T_lag):
    dynamic_od_lag.append(dynamic_od_daily[i:T_day-T_lag+i])
dynamic_od_lag = np.stack(dynamic_od_lag, axis=-1)  # (T-T_lag, N, 4, 7)

# 如果希望将 4 个通道和 7 天滞后展开成单一维度,4D->2D
T_final, N, C, L = dynamic_od_lag.shape
dynamic_od_lag_reshape = dynamic_od_lag.reshape(T_final, N, C*L)  # (T-T_lag, N, 12)

print("="*10,dynamic_od_lag_reshape.shape)

# -------------------------
# 拼接最终输入
# -------------------------
X = np.concatenate([
    spatial_expand,
    weather_expand,
    crime_lag
    # dynamic_od_lag_reshape
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
    # 1. 标准化特征
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(spatial_features)
    
    # 2. K-Means 聚类：将环境相似的网格聚在一起
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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