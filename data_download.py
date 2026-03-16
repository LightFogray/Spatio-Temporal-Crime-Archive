import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

# 设置日志
ox.settings.log_console=True
ox.settings.use_cache=True



# 获取芝加哥行政边界
place_name = "Chicago, Illinois, USA"
chicago_boundary = ox.geocode_to_gdf(place_name)

# 获取poi点数据
tags = {
    "amenity": ["restaurant", "bar", "school", "hospital", "police", "nightclub"],
    "shop": True
}

pois = ox.features_from_place(place_name, tags)

# 只保留点或多边形中心
pois = pois[pois.geometry.notnull()]

# 如果是多边形，取中心点
pois["geometry"] = pois["geometry"].centroid

# 投影到米制坐标（很重要）
pois = pois.to_crs(epsg=3857)
chicago_boundary = chicago_boundary.to_crs(epsg=3857)

# 构建规则网格 1km x 1km
# 获取边界范围
minx, miny, maxx, maxy = chicago_boundary.total_bounds

grid_size = 1000  # 1000米 = 1km

rows = int(np.ceil((maxy - miny) / grid_size))
cols = int(np.ceil((maxx - minx) / grid_size))

polygons = []
for i in range(cols):
    for j in range(rows):
        x1 = minx + i * grid_size
        y1 = miny + j * grid_size
        x2 = x1 + grid_size
        y2 = y1 + grid_size
        
        polygons.append(
            gpd.GeoSeries.box(x1, y1, x2, y2)
        )

grid = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:3857")

# 裁剪到芝加哥范围
grid = gpd.overlay(grid, chicago_boundary, how='intersection')


# 统计每个网格内的POI数量
# 空间连接
joined = gpd.sjoin(pois, grid, how="left", predicate="within")

# 添加网格ID
grid["grid_id"] = range(len(grid))

# 统计POI类型数量
poi_counts = joined.groupby(["grid_id", "amenity"]).size().unstack(fill_value=0)

# 合并到grid
grid = grid.merge(poi_counts, on="grid_id", how="left").fillna(0)

feature_cols = ["restaurant", "bar", "school", "hospital", "police", "nightclub"]

for col in feature_cols:
    if col not in grid.columns:
        grid[col] = 0

# 提取poi特征矩阵
poi_features = grid[feature_cols].values

# log归一化
poi_features = np.log1p(poi_features)

grid.to_file("chicago_grid_with_poi.shp") # 保存带有POI特征的网格数据
np.save("poi_features.npy", poi_features) # 保存POI特征矩阵


# 构建邻接矩阵
from sklearn.metrics.pairwise import euclidean_distances

# 获取网格中心点
grid["centroid"] = grid.geometry.centroid
centroids = np.array([[geom.x, geom.y] for geom in grid["centroid"]])

# 计算距离矩阵
dist_matrix = euclidean_distances(centroids)

sigma = 2000  # 2km影响范围
A = np.exp(-dist_matrix**2 / sigma**2)

# 自连接
np.fill_diagonal(A, 1)

np.save("adj_matrix.npy", A) # 保存邻接矩阵


"""
目前拥有以下数据文件：
- chicago_grid_with_poi.shp: 包含网格和POI特征的Shapefile
- poi_features.npy: 每个网格的POI特征矩阵
- adj_matrix.npy: 网格之间的邻接矩阵

接下来可以使用这些数据进行图神经网络的训练和分析，或者进行空间分析和可视化。
将犯罪数据聚合到网格中，构建犯罪特征矩阵，并与POI特征矩阵结合进行分析。
喂给STGCN模型进行训练，预测未来的犯罪热点区域。
"""
