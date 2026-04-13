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
    """
    构建与 chicago_grid 对齐的 100x100 OD 网格
    使用相同的 EPSG:3857 投影坐标系
    """
    # 读取 chicago_grid 获取边界
    grid_orig = gpd.read_file("data/processed/chicago_grid.shp")
    bounds = grid_orig.total_bounds  # [minx, miny, maxx, maxy]

    print(f"原始网格边界: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")

    # 参数
    NUM_X = 100
    NUM_Y = 100

    dx = (bounds[2] - bounds[0]) / NUM_X
    dy = (bounds[3] - bounds[1]) / NUM_Y

    geoms = []
    for i in range(NUM_X):
        for j in range(NUM_Y):
            x0 = bounds[0] + i*dx
            y0 = bounds[1] + j*dy
            x1 = x0 + dx
            y1 = y0 + dy
            geoms.append(box(x0, y0, x1, y1))

    grid_big = gpd.GeoDataFrame({"geometry": geoms}, crs=grid_orig.crs)

    # 保存
    os.makedirs("data/processed", exist_ok=True)
    grid_big.to_file("data/processed/big_grid_10000.shp")

    print(f"大网格构建完成: {len(grid_big)} 个网格")
    print(f"CRS: {grid_big.crs}")
    print(f"保存到: data/processed/big_grid_10000.shp")

build_grid_od_10000()