import geopandas as gpd
import numpy as np
import os
from shapely.geometry import box


def build_grid(grid_size=1000):

    boundary = gpd.read_file("data/raw/chicago_boundary.geojson")
    boundary = boundary.to_crs(epsg=3857)

    minx, miny, maxx, maxy = boundary.total_bounds

    rows = int(np.ceil((maxy - miny) / grid_size))
    cols = int(np.ceil((maxx - minx) / grid_size))

    polygons = []

    for i in range(cols):
        for j in range(rows):

            x1 = minx + i * grid_size
            y1 = miny + j * grid_size
            x2 = x1 + grid_size
            y2 = y1 + grid_size

            polygons.append(box(x1, y1, x2, y2))

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:3857")

    grid = gpd.overlay(grid, boundary, how="intersection")

    grid["grid_id"] = range(len(grid))

    # 自动创建目录
    os.makedirs("data/processed", exist_ok=True)

    grid.to_file("data/processed/chicago_grid.shp")

    print("网格构建完成")

def build_grid_od_10000():
    # 参数
    NUM_X = 100
    NUM_Y = 100
    LON_MIN, LON_MAX = -88.0, -87.5   # 示例范围
    LAT_MIN, LAT_MAX = 41.0, 42.0

    dx = (LON_MAX - LON_MIN) / NUM_X
    dy = (LAT_MAX - LAT_MIN) / NUM_Y

    geoms = []
    for i in range(NUM_X):
        for j in range(NUM_Y):
            x0 = LON_MIN + i*dx
            y0 = LAT_MIN + j*dy
            x1 = x0 + dx
            y1 = y0 + dy
            geoms.append(box(x0, y0, x1, y1))

    grid_big = gpd.GeoDataFrame({"geometry": geoms})
    grid_big.crs = "EPSG:3857"   # 和原始网格一致
    grid_big.to_file("big_grid_10000.shp")

# build_grid_od_10000()