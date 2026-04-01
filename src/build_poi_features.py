import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree


def build_green_features():

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    green = gpd.read_file("data/raw/chicago_green.geojson")

    green = green.to_crs(grid.crs)

    # 相交
    inter = gpd.overlay(grid, green, how="intersection")

    inter["green_area"] = inter.area

    green_area = inter.groupby("grid_id")["green_area"].sum()

    grid["grid_area"] = grid.area

    grid = grid.merge(green_area, on="grid_id", how="left")

    grid["green_area"] = grid["green_area"].fillna(0)

    grid["green_ratio"] = grid["green_area"] / grid["grid_area"]
    green_ratio = grid["green_ratio"].values.astype(np.float32)
    green_ratio = green_ratio.reshape(-1,1)
    np.save("data/processed/green_ratio.npy",
            green_ratio)

    print("绿地特征生成完成")


def build_landuse_features():
    print("Loading data...")

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    landuse = gpd.read_file("data/raw/chicago_landuse.geojson")

    print("landuse columns:", landuse.columns)

    # 坐标统一
    landuse = landuse.to_crs(grid.crs)

    # -------------------------
    # 只保留需要字段
    # -------------------------

    landuse = landuse[["landuse","geometry"]]

    # -------------------------
    # 过滤主要土地利用类型
    # -------------------------

    valid_types = [
        "residential",
        "commercial",
        "industrial",
        "retail"
    ]

    landuse = landuse[
        landuse["landuse"].isin(valid_types)
    ]

    print("Remaining landuse types:")
    print(landuse["landuse"].value_counts())

    # -------------------------
    # overlay
    # -------------------------
    
    # 过滤点状数据和线状数据
    landuse = landuse[
        landuse.geometry.type.isin(["Polygon","MultiPolygon"])
    ]
    landuse = landuse.explode(index_parts=False)
    
    intersect = gpd.overlay(grid, landuse, how="intersection")

    intersect["area"] = intersect.geometry.area

    # 自动识别 grid id
    grid_id_col = [c for c in intersect.columns if "grid_id" in c][0]
    print("grid id column:", grid_id_col)
    # -------------------------
    # 每个 grid 各类型面积
    # -------------------------
    area_sum = intersect.groupby(
        [grid_id_col,"landuse"]
    )["area"].sum().unstack(fill_value=0)
    print("landuse categories:", area_sum.columns)
    # -------------------------
    # grid面积
    # -------------------------
    grid_area = grid.set_index("grid_id").geometry.area
    # -------------------------
    # 计算比例
    # -------------------------
    landuse_ratio = area_sum.div(grid_area, axis=0).fillna(0)
    # -------------------------
    # 转 numpy
    # -------------------------
    landuse_features = landuse_ratio.values
    print("landuse ratio shape:", landuse_features.shape)

    # -------------------------
    # 计算 landuse entropy
    # -------------------------
    p = landuse_features / (
        landuse_features.sum(axis=1,keepdims=True) + 1e-9
    )

    entropy = -(p*np.log(p+1e-9)).sum(axis=1)

    entropy = entropy.reshape(-1,1)

    landuse_features = np.hstack([
        landuse_features,
        entropy
    ])

    print("final landuse features:", landuse_features.shape)

    # -------------------------
    # 保存
    # -------------------------

    np.save(
    "data/processed/landuse_features.npy",
    landuse_features.astype(np.float32)
    )

    print("Landuse features saved!")

def build_poi_features():

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    pois = gpd.read_file("data/raw/chicago_poi.geojson")

    pois = pois.to_crs(grid.crs)

    # 空间连接
    joined = gpd.sjoin(pois, grid, predicate="within")

    # 统计POI数量
    poi_counts = joined.groupby(["grid_id", "amenity"]).size().unstack(fill_value=0)

    grid = grid.merge(poi_counts, on="grid_id", how="left").fillna(0)

    feature_cols = ["restaurant", "bar", "school", "hospital", "police", "nightclub"]

    for col in feature_cols:
        if col not in grid.columns:
            grid[col] = 0

    # =====================
    # 1 POI类型特征
    # =====================

    poi_features = grid[feature_cols].values
    poi_features = np.log1p(poi_features)

    # =====================
    # 2 POI密度
    # =====================

    grid["area"] = grid.geometry.area

    grid["poi_total"] = grid[feature_cols].sum(axis=1)

    grid["poi_density"] = grid["poi_total"] / grid["area"]

    density_feature = grid["poi_density"].values.reshape(-1,1)

    # =====================
    # 3 POI多样性
    # =====================

    counts = grid[feature_cols].values

    p = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)

    entropy = -(p * np.log(p + 1e-9)).sum(axis=1)

    diversity_feature = entropy.reshape(-1,1)

    # ======================
    # 4 POI Proximity 距离特征
    # ======================

    grid_centroids = np.vstack([
        grid.geometry.centroid.x,
        grid.geometry.centroid.y
    ]).T

    proximity_features = []

    for poi_type in feature_cols:

        subset = pois[pois["amenity"] == poi_type]

        if len(subset) == 0:
            dist = np.full(len(grid), 10000)
        else:

            coords = np.vstack([
                subset.geometry.x,
                subset.geometry.y
            ]).T

            tree = BallTree(coords)

            dist, _ = tree.query(grid_centroids, k=1)

            dist = dist.flatten()

        proximity_features.append(dist)

    proximity_features = np.array(proximity_features).T

    proximity_features = np.log1p(proximity_features)


    # 加入nightlife_index夜生活指标
    grid["nightlife_index"] = (
        grid["bar"] +
        grid["nightclub"]
    )
    nightlife_feature = np.log1p(
        grid["nightlife_index"]
    ).values.reshape(-1,1)


    # =====================
    # 合并所有特征
    # =====================

    final_features = np.hstack([
        poi_features,
        density_feature,
        diversity_feature,
        proximity_features,nightlife_feature
    ])

    np.save("data/processed/poi_features.npy", final_features.astype(np.float32))

    print("POI特征生成完成")


def build_road_features():

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    roads = gpd.read_file("data/raw/chicago_roads.geojson")

    roads = roads.to_crs(grid.crs)

    intersect = gpd.overlay(roads, grid, how="intersection")

    intersect["length"] = intersect.geometry.length

    road_length = intersect.groupby("grid_id")["length"].sum()

    grid = grid.merge(road_length, on="grid_id", how="left").fillna(0)

    grid["road_density"] = grid["length"] / grid.geometry.area

    grid.to_file("data/processed/grid_roads.shp")

    print("道路特征生成完成")


import rasterio
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling


def build_nightlight_feature(target_resolution=500):

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    print(grid.crs)
    # 首先降低TIFF分辨率
    tif_path = "data/raw/Chicago_VIIRS_2025.tif"
    resampled_tif = "data/processed/Chicago_VIIRS_2025_resampled.tif"
    dst_crs = "EPSG:3857"

    with rasterio.open(tif_path) as src:
        # 计算新的变换参数
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs,
            src.width, src.height,*src.bounds,
            resolution=target_resolution
        )
        
        # 重采样
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(resampled_tif, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
    stats = zonal_stats(
        grid,
        resampled_tif,
        stats=["mean", "std"],
        geojson_out=False,
        all_touched=True,  # 只计算触及的像素
        nodata=0,  # 指定NoData值
        raster_out=False  # 不返回栅格数据
    )

    night_mean = np.array(
        [0 if s["mean"] is None else s["mean"] for s in stats],
        dtype=np.float32
    )

    night_std = np.array(
        [0 if s["std"] is None else s["std"] for s in stats],
        dtype=np.float32
    )

    features = np.vstack([night_mean, night_std]).T
    np.save("data/processed/nightlight_features.npy", features.astype(np.float32))
    print("夜间灯光特征完成")


# 生成道路密度特征
def roadshp2npy():
    grid = gpd.read_file("data/processed/grid_roads.shp")
    # print("grid columns:", grid.columns)
    # 长度限制10个字符，所以road_density被截断了
    road_density = grid["road_densi"].values
    road_density = road_density.reshape(-1,1)
    np.save("data/processed/road_density.npy", road_density.astype(np.float32))
    print("road_density shape:", road_density.shape)

import pandas as pd
def weather_features():
    # 这里可以添加天气特征的生成代码
    weather = pd.read_csv("data/external/chicago_weather_2021-2025.csv")
    weather["temp_mean"] = (
    weather["temp_max"] + weather["temp_min"]) / 2
    weather["date"] = pd.to_datetime(weather["date"])
    weather["rain_flag"] = (
        weather["rain"] > 0
    ).astype(int)
    weather["snow_flag"] = (
        weather["snow"] > 0
    ).astype(int)
    weather["month"] = weather["date"].dt.month
    weather["weekday"] = weather["date"].dt.weekday
    weather["is_weekend"] = weather["weekday"].isin([5,6]).astype(int)
    weather_features = weather[[
        "rain_flag",
        "snow_flag",
        "temp_max",
        "temp_min",
        "temp_mean",
        "month",
        "weekday",
        "is_weekend"
    ]].values

    print(weather_features.shape)
    np.save(
        "data/processed/weather_features.npy",
        weather_features.astype(np.float32)
    )

# 从卫星遥感提取的绿地覆盖特征
def build_green_remote_features():

    df = pd.read_csv("data/raw/Chicago_NDVI.csv")
    features = df[["NDVI_mean","NDVI_std"]].values
    np.save(
        "data/processed/green_remote_features.npy",
        features.astype(np.float32)
    )

def fuse_green_features():

    print("Fusing green features...")

    # OSM绿地
    osm_green = np.load(
        "data/processed/green_ratio.npy"
    )

    # 遥感绿地
    remote_green = np.load(
        "data/processed/green_remote_features.npy"
    )

    # 拼接
    green_features = np.hstack([
        osm_green,
        remote_green
    ])

    print("Final green feature shape:", green_features.shape)

    np.save(
        "data/processed/green_features.npy",
        green_features.astype(np.float32)
    )

    print("Final green features saved!")

def build_camera_features():

    grid = gpd.read_file("data/processed/chicago_grid.shp")
    camera = gpd.read_file("data/raw/cameras.geojson")

    camera = camera.to_crs(grid.crs)

    density = []
    distance = []

    for idx, cell in grid.iterrows():

        inside = camera.within(cell.geometry)

        density.append(inside.sum())

        nearest = camera.distance(cell.geometry.centroid).min()

        distance.append(nearest)

    density = np.array(density)
    distance = np.array(distance)

    density = density / density.max()
    distance = distance / distance.max()

    features = np.stack([density, distance], axis=1)
    np.save("data/processed/camera_features.npy", features.astype(np.float32))

    print("摄像头特征生成完成")
    print("shape:", features.shape)

# build_nightlight_feature()
# build_road_features()
# roadshp2npy()
# build_landuse_features()
# build_poi_features()
# build_green_features()
# weather_features()
# build_green_remote_features()
# fuse_green_features()
# build_camera_features()