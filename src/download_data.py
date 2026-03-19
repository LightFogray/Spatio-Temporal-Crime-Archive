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

# download_landuse()

import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
from sodapy import Socrata  # 用于访问 Chicago Data Portal

# =================配置区域=================
# 1. Divvy 数据配置
DIVvy_BASE_URL = "https://divvy-tripdata.s3.amazonaws.com"
DOWNLOAD_DIR = "chicago_data"
DIVVY_SAVE_DIR = os.path.join(DOWNLOAD_DIR, "divvy")
BUS_SAVE_DIR = os.path.join(DOWNLOAD_DIR, "cta_bus")

# 定义需要下载的年份和月份
YEARS = [2021, 2022, 2023, 2024, 2025]
# 2025年可能只有部分月份，可以根据实际情况调整
MONTHS_2025 = [1] # 假设只到3月，可根据实际更新

# 2. CTA Bus 数据配置 (Chicago Data Portal)
# 数据集 ID: wrvz-psew (这是 Taxi 的 ID, 需要替换为 Bus 的 ID)
# 经过搜索，CTA Bus Ridership 的数据集 ID 是: bmxz-8z8h (CTA Ridership - Bus Routes-Stops-Daily Totals by Direction)
# 注意：如果 ID 变更，请在 data.cityofchicago.org 上查找最新 ID
CTA_BUS_DATASET_ID = "bmxz-8z8h" 
CHICAGO_DOMAIN = "data.cityofchicago.org"

# 创建目录
os.makedirs(DIVVY_SAVE_DIR, exist_ok=True)
os.makedirs(BUS_SAVE_DIR, exist_ok=True)

# =================函数定义=================

def download_divvy_data():
    print("🚀 开始下载 Divvy 共享单车数据...")
    
    files_to_download = []
    for year in YEARS:
        if year == 2025:
            months = MONTHS_2025
        else:
            months = range(1, 13)
        
        for month in months:
            filename = f"{year}{month:02d}-divvy-tripdata.zip"
            url = f"{DIVvy_BASE_URL}/{filename}"
            save_path = os.path.join(DIVVY_SAVE_DIR, filename)
            files_to_download.append((url, save_path))

    for url, save_path in tqdm(files_to_download, desc="下载 Divvy 文件"):
        if os.path.exists(save_path):
            print(f"跳过已存在: {os.path.basename(save_path)}")
            continue
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ 下载成功: {os.path.basename(save_path)}")
            
        except Exception as e:
            print(f"❌ 下载失败 {os.path.basename(save_path)}: {e}")

    # 自动解压
    print("\n📦 开始解压 Divvy 文件...")
    for zip_file in tqdm(os.listdir(DIVVY_SAVE_DIR), desc="解压"):
        if zip_file.endswith('.zip'):
            zip_path = os.path.join(DIVVY_SAVE_DIR, zip_file)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # 解压到同名文件夹
                    extract_dir = os.path.join(DIVVY_SAVE_DIR, zip_file.replace('.zip', ''))
                    os.makedirs(extract_dir, exist_ok=True)
                    zip_ref.extractall(extract_dir)
                # 可选：删除 zip 包节省空间
                # os.remove(zip_path) 
            except Exception as e:
                print(f"❌ 解压失败 {zip_file}: {e}")
    
    print("✅ Divvy 数据处理完成！")

def download_cta_bus_data():
    print("\n🚌 开始下载 CTA 公交车客流数据...")
    
    # 初始化 Socrata 客户端 (无需 token 即可访问公开数据，但有 token 限制更高)
    # 你可以在 data.cityofchicago.org 申请一个免费的 App Token
    client = Socrata(CHICAGO_DOMAIN, None) # 填入你的 App Token 代替 None
    
    # 设定时间范围 (示例：获取 2021-01-01 到 2025-12-31)
    # 注意：一次性拉取 5 年数据可能非常大，建议分年或分月拉取
    # 这里演示拉取全部，如果超时请分段
    
    print("正在查询数据，这可能需要几分钟...")
    try:
        # 查询所有数据，按日期排序
        # 字段名参考：date, route_id, stop_id, direction, rides
        results = client.get(CTA_BUS_DATASET_ID, 
                             order="date", 
                             limit=5000000) # 设置一个足够大的 limit
        
        if not results:
            print("⚠️ 未获取到数据，请检查数据集 ID 或网络连接。")
            return

        df = pd.DataFrame.from_records(results)
        
        # 保存为 CSV
        output_file = os.path.join(BUS_SAVE_DIR, "cta_bus_ridership_2021_2025.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ CTA 公交数据已保存至: {output_file}")
        print(f"   总记录数: {len(df)}")
        
    except Exception as e:
        print(f"❌ 下载 CTA 数据失败: {e}")
        print("💡 提示：如果数据量太大，请尝试在 client.get() 中添加 where 条件分段下载，例如：where=\"date between '2021-01-01' and '2021-12-31'\"")

if __name__ == "__main__":
    # 1. 下载 Divvy
    download_divvy_data()
    
    # 2. 下载 CTA Bus
    # 注意：首次运行可能需要安装 sodapy: pip install sodapy
    try:
        download_cta_bus_data()
    except ImportError:
        print("\n⚠️ 缺少 sodapy 库，无法下载公交数据。请运行: pip install sodapy")
        print("   或者您可以手动去 https://data.cityofchicago.org/ 下载 CSV 文件。")
    
    print("\n🎉 所有任务完成！")

    