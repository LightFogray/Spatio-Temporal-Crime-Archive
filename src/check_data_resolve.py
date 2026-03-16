import geopandas as gpd
import pandas as pd



grid = gpd.read_file("../data/processed/chicago_grid.shp")
pois = gpd.read_file("../data/raw/chicago_poi.geojson")
crime = pd.read_csv("../data/raw/chicago_violence_data_2021-2025.csv")
roads = gpd.read_file("../data/raw/chicago_roads.geojson")
landuse = gpd.read_file("../data/raw/chicago_landuse.geojson")
green = gpd.read_file("../data/raw/chicago_green.geojson")

# print(grid.crs)
# print(grid.shape)
# print(grid.total_bounds)
# print(grid.area.mean())
# print(grid.head())

print("\n==================================\n")
print(grid.crs)
print(pois.crs)
print(roads.crs)
print(landuse.crs)
print(green.crs)

print("\n==================================\n")
print(pois.geom_type.unique())
print(roads.geom_type.unique())
print(landuse.geom_type.unique())
print(green.geom_type.unique())


# print(pois.crs)
# print(pois.shape)
# print(pois.columns)
# print(pois.head())

# 与边界数据的坐标不一致，需要转换
pois = pois.to_crs(grid.crs)
# print(pois.crs)


crime_gdf = gpd.GeoDataFrame(
    crime,
    geometry=gpd.points_from_xy(
        crime.Longitude,
        crime.Latitude
    ),
    crs="EPSG:4326"
)
crime_gdf = crime_gdf.to_crs("EPSG:3857")

# 将犯罪点与网格进行空间连接，统计每个网格内的犯罪数量
crime_grid = gpd.sjoin(
    crime_gdf,
    grid,
    predicate="within"
)
crime_count = crime_grid.groupby("grid_id").size()
# print(crime_gdf.total_bounds)
# print(grid.total_bounds)
# print(crime_count)