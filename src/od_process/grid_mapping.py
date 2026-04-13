import numpy as np
import geopandas as gpd

print("="*60)
print("网格映射 - 坐标系转换版")
print("="*60)

grid_orig = gpd.read_file("data/processed/chicago_grid.shp")
grid_big = gpd.read_file("data/processed/big_grid_10000.shp")

print(f"\n原始网格 (crime): {len(grid_orig)} 个")
print(f"大网格 (OD): {len(grid_big)} 个")

print(f"\n原始网格 CRS: {grid_orig.crs}")
print(f"大网格 CRS: {grid_big.crs}")

# 检查坐标范围判断坐标系
bounds_orig = grid_orig.total_bounds
bounds_big = grid_big.total_bounds

print(f"\n原始网格边界: Lon [{bounds_orig[0]:.2f}, {bounds_orig[2]:.2f}], Lat [{bounds_orig[1]:.2f}, {bounds_orig[3]:.2f}]")
print(f"大网格边界: Lon [{bounds_big[0]:.2f}, {bounds_big[2]:.2f}], Lat [{bounds_big[1]:.2f}, {bounds_big[3]:.2f}]")

# 判断坐标系并转换
# 如果原始网格数值很大（百万级），是投影坐标系（EPSG:3857）
# 如果大网格数值很小（-88到-87），是地理坐标系（EPSG:4326）

if abs(bounds_orig[0]) > 1000:  # 投影坐标系
    print("\n检测到原始网格是投影坐标系 (EPSG:3857 Web Mercator)")
    print("需要将大网格转换到同一坐标系...")

    # 将大网格从 WGS84 转换到 Web Mercator
    grid_big = grid_big.to_crs("EPSG:3857")
    print(f"转换后大网格 CRS: {grid_big.crs}")

    bounds_big_new = grid_big.total_bounds
    print(f"转换后大网格边界: [{bounds_big_new[0]:.2f}, {bounds_big_new[2]:.2f}], [{bounds_big_new[1]:.2f}, {bounds_big_new[3]:.2f}]")

elif abs(bounds_big[0]) > 1000:  # 反过来
    print("\n检测到大网格是投影坐标系")
    print("将原始网格转换到同一坐标系...")
    grid_orig = grid_orig.to_crs(grid_big.crs)
else:
    print("\n警告: 两个网格都在地理坐标系，需要手动指定目标投影坐标系")
    # 假设原始数据是芝加哥，使用 EPSG:3857
    grid_orig = grid_orig.to_crs("EPSG:3857")
    grid_big = grid_big.to_crs("EPSG:3857")

# 检查转换后的重叠
bounds_orig = grid_orig.total_bounds
bounds_big = grid_big.total_bounds

print(f"\n转换后边界对比:")
print(f"  原始网格: Lon [{bounds_orig[0]:.2f}, {bounds_orig[2]:.2f}], Lat [{bounds_orig[1]:.2f}, {bounds_orig[3]:.2f}]")
print(f"  大网格:   Lon [{bounds_big[0]:.2f}, {bounds_big[2]:.2f}], Lat [{bounds_big[1]:.2f}, {bounds_big[3]:.2f}]")

# 计算重叠
overlap_lon = max(0, min(bounds_orig[2], bounds_big[2]) - max(bounds_orig[0], bounds_big[0]))
overlap_lat = max(0, min(bounds_orig[3], bounds_big[3]) - max(bounds_orig[1], bounds_big[1]))
print(f"\n重叠区域: Lon {overlap_lon:.2f} m × Lat {overlap_lat:.2f} m")

if overlap_lon == 0 or overlap_lat == 0:
    print("❌ 转换后仍然没有重叠！")
    exit(1)

# 构建映射表
print("\n" + "="*60)
print("构建映射表...")
print("="*60)

# 方法1: 使用 contains（大网格中心点在哪个原始网格内）
mapping = []
for idx_big, geom_big in enumerate(grid_big.geometry):
    centroid = geom_big.centroid
    found = grid_orig[grid_orig.contains(centroid)]
    if len(found) > 0:
        mapping.append(found.index[0])
    else:
        mapping.append(-1)

mapping = np.array(mapping)
n_mapped = (mapping != -1).sum()
print(f"使用 contains 后映射率: {n_mapped}/{len(mapping)} ({n_mapped/len(mapping)*100:.1f}%)")

# 如果 contains 效果不好，使用 intersects
if n_mapped < len(mapping) * 0.5:
    print("\n尝试使用 intersects...")
    mapping = []
    for idx_big, geom_big in enumerate(grid_big.geometry):
        found = grid_orig[grid_orig.intersects(geom_big)]
        if len(found) > 0:
            # 选择交集面积最大的
            intersections = found.geometry.intersection(geom_big)
            areas = intersections.area
            best_idx = areas.idxmax()
            mapping.append(best_idx)
        else:
            mapping.append(-1)

    mapping = np.array(mapping)
    n_mapped = (mapping != -1).sum()
    print(f"使用 intersects 后映射率: {n_mapped}/{len(mapping)} ({n_mapped/len(mapping)*100:.1f}%)")

# 如果还是不好，使用最近距离
if n_mapped < len(mapping) * 0.5:
    print("\n尝试最近距离匹配...")
    grid_orig_centroids = grid_orig.geometry.centroid
    grid_big_centroids = grid_big.geometry.centroid

    mapping = []
    for idx_big, centroid_big in enumerate(grid_big_centroids):
        distances = grid_orig_centroids.distance(centroid_big)
        min_dist = distances.min()
        min_idx = distances.idxmin()

        # 阈值 500 米
        if min_dist < 500:
            mapping.append(min_idx)
        else:
            mapping.append(-1)

    mapping = np.array(mapping)
    n_mapped = (mapping != -1).sum()
    print(f"最近距离匹配后映射率: {n_mapped}/{len(mapping)} ({n_mapped/len(mapping)*100:.1f}%)")

# 检查映射分布
unique_targets = len(set(m for m in mapping if m != -1))
print(f"映射到的唯一网格: {unique_targets}/{len(grid_orig)}")

# 聚合数据
print("\n" + "="*60)
print("聚合数据")
print("="*60)

if n_mapped == 0:
    print("❌ 映射失败，无法聚合数据")
    exit(1)

bike_inflow_10000 = np.load("data/processed/bike_inflow_daily.npy")
T = bike_inflow_10000.shape[0]
N_orig = len(grid_orig)

bike_inflow_1246 = np.zeros((T, N_orig), dtype=np.float32)
bike_outflow_1246 = np.zeros((T, N_orig), dtype=np.float32)
taxi_inflow_1246 = np.zeros((T, N_orig), dtype=np.float32)
taxi_outflow_1246 = np.zeros((T, N_orig), dtype=np.float32)

bike_outflow_10000 = np.load("data/processed/bike_outflow_daily.npy")
taxi_inflow_10000 = np.load("data/processed/taxi_inflow_daily.npy")
taxi_outflow_10000 = np.load("data/processed/taxi_outflow_daily.npy")

for i_big, i_orig in enumerate(mapping):
    if i_orig == -1:
        continue
    bike_inflow_1246[:, i_orig] += bike_inflow_10000[:, i_big]
    bike_outflow_1246[:, i_orig] += bike_outflow_10000[:, i_big]
    taxi_inflow_1246[:, i_orig] += taxi_inflow_10000[:, i_big]
    taxi_outflow_1246[:, i_orig] += taxi_outflow_10000[:, i_big]

print(f"\n聚合后统计:")
print(f"  bike_inflow_1246:  非零={(bike_inflow_1246 > 0).sum()}, 最大值={bike_inflow_1246.max():.2f}")
print(f"  bike_outflow_1246: 非零={(bike_outflow_1246 > 0).sum()}, 最大值={bike_outflow_1246.max():.2f}")
print(f"  taxi_inflow_1246:  非零={(taxi_inflow_1246 > 0).sum()}, 最大值={taxi_inflow_1246.max():.2f}")
print(f"  taxi_outflow_1246: 非零={(taxi_outflow_1246 > 0).sum()}, 最大值={taxi_outflow_1246.max():.2f}")

# 保存
if (bike_inflow_1246 > 0).sum() > 0:
    np.save("data/processed/bike_inflow_1246.npy", bike_inflow_1246)
    np.save("data/processed/taxi_inflow_1246.npy", taxi_inflow_1246)
    np.save("data/processed/bike_outflow_1246.npy", bike_outflow_1246)
    np.save("data/processed/taxi_outflow_1246.npy", taxi_outflow_1246)
    print("\n✅ 保存成功!")
else:
    print("\n❌ 警告: 聚合后数据全为0!")

print("\n" + "="*60)
print("完成")
print("="*60)
