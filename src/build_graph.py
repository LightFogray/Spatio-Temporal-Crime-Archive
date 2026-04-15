import geopandas as gpd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


# 读取grid数据
grid = gpd.read_file("data/processed/chicago_grid.shp")
N = len(grid)
print("grid number:", N)

# 构建空间邻接图
def build_adj_adaptive():
    A_spatial = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if grid.geometry[i].touches(grid.geometry[j]):
                A_spatial[i, j] = 1

    poi = np.load("data/processed/poi_features.npy")
    landuse = np.load("data/processed/landuse_features.npy")
    night = np.load("data/processed/nightlight_features.npy")
    road = np.load("data/processed/road_density.npy")
    green = np.load("data/processed/green_features.npy")
    green_squared = green ** 2  # 二次项 体现双向作用
    feature = np.hstack([
        poi,
        landuse,
        night,
        road,
        green,
        green_squared
    ])

    # 检查并处理 NaN
    nan_count = np.isnan(feature).sum()
    if nan_count > 0:
        print(f"  Warning: Feature contains {nan_count} NaN values, filling with 0")
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算特征相似度
    S_feature = cosine_similarity(feature)
    A_adaptive = A_spatial * S_feature
    A_adaptive = A_adaptive.astype(np.float32)
    np.save("data/processed/adj_adaptive.npy", A_adaptive)
    print("adj_adaptive saved, shape:", A_adaptive.shape)
    return A_adaptive


# 距离衰减图
def build_adj_distance():
    coords = np.array([
        [geom.centroid.x, geom.centroid.y]
        for geom in grid.geometry
    ])
    dist = euclidean_distances(coords)
    sigma = dist.mean()
    A_distance = np.exp(-(dist**2) / (sigma**2))
    A_distance = A_distance.astype(np.float32)
    np.save("data/processed/adj_distance.npy", A_distance)
    print("adj_distance saved, shape:", A_distance.shape)
    return A_distance


# ==================== 双犯罪传播图 ====================

def build_dual_crime_graphs(window=30, sigma=None, use_spatial_mask=True):
    """
    分别构建暴力犯罪和财产犯罪的传播图（支持稀疏优化）

    Args:
        window: 时间窗口大小
        sigma: 高斯核带宽
        use_spatial_mask: 是否使用空间邻近图作为掩码进行稀疏化

    Returns:
        A_violent: 暴力犯罪传播图
        A_property: 财产犯罪传播图
    """
    crime = np.load("data/processed/crime_combined_timeseries.npy")  # (T, N, 2)
    T, N, C = crime.shape

    print(f"Building dual crime graphs from crime shape: {crime.shape}")
    print(f"  Channel 0: Violent Crime")
    print(f"  Channel 1: Property Crime")
    print(f"  Sparse optimization: {use_spatial_mask}")

    # 加载空间邻近图作为掩码
    spatial_mask = None
    if use_spatial_mask:
        try:
            A_spatial = np.load("data/processed/adj_adaptive.npy")  # (N, N)
            # 转换为布尔掩码（只保留有空间连接的边）
            spatial_mask = (A_spatial > 0)
            avg_degree = spatial_mask.sum(axis=1).mean()
            print(f"  Spatial mask loaded: avg_degree={avg_degree:.2f}")
            print(f"  Computation reduction: {N**2} -> {int(N * avg_degree)} ({N**2 / (N * avg_degree):.1f}x)")
        except FileNotFoundError:
            print("  Warning: Spatial mask not found, using dense computation")
            use_spatial_mask = False

    def compute_gaussian_similarity_sparse(crime_channel, window, sigma, mask):
        """
        稀疏高斯相似度计算
        只计算 mask[i, j] == True 的位置
        """
        A_list = []
        T_steps = crime_channel.shape[0]
        N = crime_channel.shape[1]

        # 获取所有需要计算的边 (i, j)
        edges = np.argwhere(mask)  # (E, 2)
        print(f"  Computing {len(edges)} edges out of {N**2} possible")

        for t in range(window, T_steps):
            crime_window = crime_channel[t-window:t].T  # (N, window)

            # 计算 sigma
            if sigma is None:
                sigma_val = np.std(crime_window)
                if sigma_val == 0:
                    sigma_val = 1.0
            else:
                sigma_val = sigma

            # 初始化稀疏矩阵
            sim = np.zeros((N, N), dtype=np.float32)

            # 只对邻居计算相似度
            for i, j in edges:
                diff = crime_window[i] - crime_window[j]  # (window,)
                dist2 = np.sum(diff ** 2)
                sim[i, j] = np.exp(-dist2 / (2 * sigma_val ** 2))

            # 对称化
            sim = (sim + sim.T) / 2
            A_list.append(sim)

        return np.array(A_list, dtype=np.float32)

    def compute_gaussian_similarity_dense(crime_channel, window, sigma):
        """稠密高斯相似度计算（原始方法）"""
        A_list = []
        T_steps = crime_channel.shape[0]

        for t in range(window, T_steps):
            crime_window = crime_channel[t-window:t].T  # (N, window)

            if sigma is None:
                sigma_val = np.std(crime_window)
                if sigma_val == 0:
                    sigma_val = 1.0
            else:
                sigma_val = sigma

            # pairwise squared distance
            diff = crime_window[:, np.newaxis, :] - crime_window[np.newaxis, :, :]
            dist2 = np.sum(diff**2, axis=2)

            # Gaussian similarity
            sim = np.exp(-dist2 / (2 * sigma_val**2))
            sim[sim < 0] = 0.0

            A_list.append(sim.astype(np.float32))

        return np.array(A_list, dtype=np.float32)

    # 分别计算两张图
    crime_violent = crime[:, :, 0]  # (T, N)
    crime_property = crime[:, :, 1]  # (T, N)

    print("\nBuilding violent crime graph...")
    if use_spatial_mask and spatial_mask is not None:
        A_violent = compute_gaussian_similarity_sparse(crime_violent, window, sigma, spatial_mask)
    else:
        A_violent = compute_gaussian_similarity_dense(crime_violent, window, sigma)

    print("Building property crime graph...")
    if use_spatial_mask and spatial_mask is not None:
        A_property = compute_gaussian_similarity_sparse(crime_property, window, sigma, spatial_mask)
    else:
        A_property = compute_gaussian_similarity_dense(crime_property, window, sigma)

    # 保存
    np.save("data/processed/adj_crime_violent.npy", A_violent)
    np.save("data/processed/adj_crime_property.npy", A_property)

    print(f"\nViolent crime graph saved: {A_violent.shape}")
    print(f"Property crime graph saved: {A_property.shape}")
    if use_spatial_mask:
        print(f"  Sparse format: non-zero ~{(A_violent[0] > 0).sum() / A_violent[0].size * 100:.2f}%")

    return A_violent, A_property


# ==================== OD 流图构建 ====================

def build_od_graph(K=10, threshold=0.3):
    """
    构建OD流功能相似图

    基于OD流特征的功能相似性（人流模式相似的区域）

    Args:
        K: Top-K 稀疏化参数
        threshold: 相似度阈值

    Returns:
        A_od: (N, N) OD功能相似图
    """
    od_flow = np.load("data/processed/dynamic_od_flow.npy")  # (T, N, 4)
    T, N, C = od_flow.shape

    print(f"\nBuilding OD graph from OD flow shape: {od_flow.shape}")
    print(f"  Channels: bike_in, bike_out, taxi_in, taxi_out")

    # 时间平均得到每个网格的OD特征
    od_features = od_flow.mean(axis=0)  # (N, 4)

    # 检查并处理 NaN
    nan_count = np.isnan(od_features).sum()
    if nan_count > 0:
        print(f"  Warning: OD features contain {nan_count} NaN values, filling with 0")
        od_features = np.nan_to_num(od_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算余弦相似度
    similarity = cosine_similarity(od_features)  # (N, N)

    print(f"  OD similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")

    # Top-K 稀疏化
    A_od = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # 找到相似度最高的K个邻居（包括自己）
        top_k_idx = np.argsort(similarity[i])[-K:]

        for idx in top_k_idx:
            if similarity[i, idx] > threshold:
                A_od[i, idx] = similarity[i, idx]

    # 对称化
    A_od = (A_od + A_od.T) / 2

    # 归一化（行归一化）
    row_sums = A_od.sum(axis=1, keepdims=True)
    A_od = np.divide(A_od, row_sums, where=row_sums!=0)

    # 保存
    np.save("data/processed/adj_od.npy", A_od.astype(np.float32))

    print(f"OD graph saved: {A_od.shape}")
    print(f"  Non-zero elements: {(A_od > 0).sum()}/{A_od.size} ({(A_od > 0).mean()*100:.2f}%)")
    print(f"  Average degree: {(A_od > 0).sum(axis=1).mean():.2f}")

    return A_od


# ==================== 主流程 ====================

def build_all_graphs():
    """构建所有图结构"""
    print("="*60)
    print("Building All Graph Structures")
    print("="*60)

    # 1. 基础空间图
    print("\n[1/5] Building spatial adaptive graph...")
    build_adj_adaptive()

    # 2. 距离衰减图
    print("\n[2/5] Building distance decay graph...")
    build_adj_distance()

    # 3. 双犯罪传播图
    print("\n[3/5] Building dual crime diffusion graphs...")
    build_dual_crime_graphs(window=30, sigma=None)

    # 4. OD流图
    print("\n[4/5] Building OD flow graph...")
    build_od_graph(K=10, threshold=0.3)

    # 5. 超图（如果存在）
    print("\n[5/5] Checking hypergraph...")
    if not os.path.exists("data/processed/adj_hypergraph.npy"):
        print("  Hypergraph not found. Run build_stgcn_input.py to generate it.")
    else:
        print("  Hypergraph exists.")

    print("\n" + "="*60)
    print("All graphs built successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - adj_adaptive.npy (Spatial connectivity)")
    print("  - adj_distance.npy (Distance decay)")
    print("  - adj_crime_violent.npy (Violent crime diffusion)")
    print("  - adj_crime_property.npy (Property crime diffusion)")
    print("  - adj_od.npy (OD flow functional similarity)")


import os
if __name__ == "__main__":
    build_all_graphs()
