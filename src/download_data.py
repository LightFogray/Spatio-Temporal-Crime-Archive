import os
import osmnx as ox
import geopandas as gpd

def download_greenland():

    place = "Chicago, Illinois, USA"

    tags = {
        "leisure": ["park"],
        "landuse": ["grass", "forest"],
        "natural": ["wood", "grassland"]
    }

    green = ox.features_from_place(place, tags)

    green = green[green.geometry.notnull()]

    # 只保留面
    green = green[
        green.geometry.type.isin(["Polygon", "MultiPolygon"])
    ]

    green.to_file("data/raw/chicago_green.geojson")

    print("绿地数据下载完成")

def download_landuse():

    place = "Chicago, Illinois, USA"

    tags = {
        "landuse": [
            "residential",
            "commercial",
            "industrial",
            "retail"
        ]
    }

    landuse = ox.features_from_place(place, tags)

    landuse = landuse[landuse.geometry.notnull()]

    landuse.to_file("data/raw/chicago_landuse.geojson")

    print("土地利用数据下载完成")

def download_chicago_data():

    os.makedirs("data/raw", exist_ok=True)

    place_name = "Chicago, Illinois, USA"

    # 获取边界
    boundary = ox.geocode_to_gdf(place_name)
    boundary.to_file("data/raw/chicago_boundary.geojson")

    tags = {
        "amenity": ["restaurant", "bar", "school", "hospital", "police", "nightclub"],
        "shop": True
    }

    pois = ox.features_from_place(place_name, tags)

    pois = pois[pois.geometry.notnull()]
    pois["geometry"] = pois.geometry.centroid

    pois.to_file("data/raw/chicago_poi.geojson")

    print("POI数据下载完成")


def download_road_network():
    place = "Chicago, Illinois, USA"
    G = ox.graph_from_place(place, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges.to_file("data/raw/chicago_roads.geojson")
    print("道路网络下载完成")

download_landuse()
    