# 出租车数据表头
# Trip ID,Taxi ID,Trip Start Timestamp,Trip End Timestamp,Trip Seconds,Trip Miles,Pickup Census Tract,Dropoff Census Tract,Pickup Community Area,Dropoff Community Area,Fare,Tips,Tolls,Extras,Trip Total,Payment Type,Company,Pickup Centroid Latitude,Pickup Centroid Longitude,Pickup Centroid Location,Dropoff Centroid Latitude,Dropoff Centroid Longitude,Dropoff Centroid  Location

# 单车数据表头
# ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual

import pandas as pd
import os

# ==============================
# 1. 参数设置
# ==============================
# 输入文件路径
input_files = {
    "bike": "./OD_data/divvy/combined_divvy_trips_data_20210101-20250101.csv",
    "taxi": "./OD_data/taxi/combined_taxi_trips_data_20210101-20250101.csv"
}

# 输出干净文件路径
output_files = {
    "bike": "./OD_data/divvy_clean.csv",
    "taxi": "./OD_data/taxi_clean.csv"
}

# 需要保留的列
columns_to_keep = {
    "bike": ["start_lng", "start_lat", "end_lng", "end_lat", "started_at"],
    "taxi": ["Pickup Centroid Longitude", "Pickup Centroid Latitude",
             "Dropoff Centroid Longitude", "Dropoff Centroid Latitude",
             "Trip Start Timestamp"]
}

# ==============================
# 2. 清洗函数
# ==============================
def clean_csv(input_path, output_path, columns, chunk_size=500_000):
    print(f"🧹 处理文件: {input_path}")
    
    # 如果文件非常大，用分块处理
    chunks = pd.read_csv(input_path, usecols=columns, chunksize=chunk_size, low_memory=False)
    cleaned_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"  ➡ 处理块 {i+1}")
        
        # 1. 去掉空值
        chunk = chunk.dropna(subset=columns)
        
        # 2. 清理数字列（去掉逗号）
        for col in chunk.columns:
            if chunk[col].dtype == object and col != columns[-1]:  # 时间列最后一列不处理
                chunk[col] = chunk[col].str.replace(",", "", regex=False)
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        
        # 3. 再次去掉无法解析的数字行
        chunk = chunk.dropna(subset=columns[:-1])
        
        cleaned_chunks.append(chunk)
    
    # 合并所有块并保存
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ 清洗完成，保存到 {output_path}, 共 {len(cleaned_df)} 行\n")
    return cleaned_df

# ==============================
# 3. 主流程
# ==============================
if __name__ == "__main__":
    for key in input_files:
        clean_csv(
            input_path=input_files[key],
            output_path=output_files[key],
            columns=columns_to_keep[key]
        )