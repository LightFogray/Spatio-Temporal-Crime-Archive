import pandas as pd
import geopandas as gpd
import numpy as np

print("Loading data")

crime = pd.read_csv("data/raw/chicago_violence_data_2021-2025.csv")

grid = gpd.read_file("data/processed/chicago_grid.shp")

# 时间处理
crime["Date"] = pd.to_datetime(
    crime["Date"],
    errors="coerce"
)

crime = crime.dropna(subset=["Date"])

crime["date"] = crime["Date"].dt.floor("D")

# 删除无坐标
crime = crime.dropna(subset=["Latitude","Longitude"])

# 转geodataframe
crime_gdf = gpd.GeoDataFrame(
    crime,
    geometry=gpd.points_from_xy(
        crime["Longitude"],
        crime["Latitude"]
    ),
    crs="EPSG:4326"
)

crime_gdf = crime_gdf.to_crs(grid.crs)

# 空间匹配
crime_join = gpd.sjoin(
    crime_gdf,
    grid[["grid_id","geometry"]],
    how="left",
    predicate="within"
)

# 统计
crime_matrix = crime_join.groupby(
    ["date","grid_id"]
).size().unstack(fill_value=0)

# 所有grid id
all_grids = grid["grid_id"].values

# 时间补齐
date_range = pd.date_range(
    crime_matrix.index.min(),
    crime_matrix.index.max()
)
crime_matrix = crime_matrix.reindex(
    date_range,
    columns=all_grids,
    fill_value=0
)


# 转numpy
crime_array = crime_matrix.values.astype(np.float32)

print("crime matrix:", crime_array.shape)

np.save(
    "data/processed/crime_grid_timeseries.npy",
    crime_array
)

print("crime timeseries saved")