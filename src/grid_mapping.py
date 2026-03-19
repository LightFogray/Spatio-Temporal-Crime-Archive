import numpy as np
import geopandas as gpd
grid_orig = gpd.read_file("data/processed/chicago_grid.shp")
grid_big = gpd.read_file("data/processed/big_grid_10000.shp")

# 读取大网格 npy
bike_inflow_10000 = np.load("data/processed/bike_inflow.npy")  # shape (T,10000)
bike_outflow_10000 = np.load("data/processed/bike_outflow.npy")

taxi_inflow_10000 = np.load("data/processed/taxi_inflow.npy")  # shape (T,10000)
taxi_outflow_10000 = np.load("data/processed/taxi_outflow.npy")

T, _ = bike_inflow_10000.shape
A, _ = taxi_inflow_10000.shape
N_orig = len(grid_orig)

# 映射表
mapping = []
for idx_big, geom in enumerate(grid_big.geometry):
    centroid = geom.centroid
    found = grid_orig[grid_orig.contains(centroid)]
    if len(found) > 0:
        mapping.append(found.index[0])
    else:
        mapping.append(-1)  # 没找到，可能在边界外

mapping = np.array(mapping)

# 聚合到原始网格
bike_inflow_1246 = np.zeros((T, N_orig), dtype=np.float32)
taxi_inflow_1246 = np.zeros((A, N_orig), dtype=np.float32)
bike_outflow_1246 = np.zeros((T, N_orig), dtype=np.float32)
taxi_outflow_1246 = np.zeros((A, N_orig), dtype=np.float32)

for i_big, i_orig in enumerate(mapping):
    if i_orig == -1:
        continue
    bike_inflow_1246[:, i_orig] += bike_inflow_10000[:, i_big]
    taxi_inflow_1246[:, i_orig] += taxi_inflow_10000[:, i_big]
    bike_outflow_1246[:, i_orig] += bike_outflow_10000[:, i_big]
    taxi_outflow_1246[:, i_orig] += taxi_outflow_10000[:, i_big]

# 保存最终 npy
np.save("data/processed/bike_inflow_1246.npy", bike_inflow_1246)
np.save("data/processed/taxi_inflow_1246.npy", taxi_inflow_1246)
np.save("data/processed/bike_outflow_1246.npy", bike_outflow_1246)
np.save("data/processed/taxi_outflow_1246.npy", taxi_outflow_1246)