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