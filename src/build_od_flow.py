import polars as pl
import numpy as np
import pandas as pd

# ==============================
# 参数
# ==============================
GRID_SIZE = 0.01
LON_MIN, LAT_MIN = -88, 41
NUM_X, NUM_Y = 100, 100
N = NUM_X * NUM_Y

file_paths = {
    "bike": "./OD_data/divvy_clean.csv",
    "taxi": "./OD_data/taxi_clean.csv"
}

columns_map = {
    "bike": ("start_lng", "start_lat", "end_lng", "end_lat", "started_at"),
    "taxi": ("Pickup Centroid Longitude", "Pickup Centroid Latitude",
             "Dropoff Centroid Longitude", "Dropoff Centroid Latitude",
             "Trip Start Timestamp")
}

# ==============================
# 坐标 → grid id
# ==============================
def coord_to_grid_polars(df, lon_col, lat_col, new_col):
    df = df.with_columns([
        ((pl.col(lon_col) - LON_MIN) / GRID_SIZE).cast(pl.Int32).alias("x"),
        ((pl.col(lat_col) - LAT_MIN) / GRID_SIZE).cast(pl.Int32).alias("y")
    ])
    df = df.with_columns(
        (pl.col("x") + pl.col("y") * NUM_X).alias(new_col)
    ).drop(["x", "y"])
    return df

# ==============================
# 时间 → 小时
# ==============================
def time_to_bin_polars(df, time_col, new_col_name):
    if time_col == "Trip Start Timestamp":  # 出租车数据
        df = df.with_columns(
            pl.col(time_col)
              .str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p", strict=False)
              .dt.truncate("1h")
              .alias(new_col_name)
        )
    else:  # 自行车或其他数据，Polars 默认解析即可
        df = df.with_columns(
            pl.col(time_col)
              .str.strptime(pl.Datetime, strict=False)
              .dt.truncate("1h")
              .alias(new_col_name)
        )
    return df

# ==============================
# 构建 OD 流
# ==============================
def build_flow_numpy_polars(file_path, cols, prefix):
    print(f"📍 Processing {prefix}...")

    df = pl.read_csv(file_path, columns=list(cols))

    # 丢掉空值
    df = df.drop_nulls()

    # 网格 ID
    df = coord_to_grid_polars(df, cols[0], cols[1], "origin_grid")
    df = coord_to_grid_polars(df, cols[2], cols[3], "dest_grid")

    # 丢掉非法网格
    df = df.filter(
        (pl.col("origin_grid") >= 0) & (pl.col("origin_grid") < N) &
        (pl.col("dest_grid") >= 0) & (pl.col("dest_grid") < N)
    )

    # 时间
    df = time_to_bin_polars(df, cols[4], "hour")
    df = df.drop_nulls(subset=["hour"])

    # 转 pandas 方便生成连续时间序列
    df_pd = df.to_pandas()
    df_pd["hour"] = pd.to_datetime(df_pd["hour"])

    min_time, max_time = df_pd["hour"].min(), df_pd["hour"].max()
    all_hours = pd.date_range(min_time, max_time, freq="H")
    hour_to_idx = {h: i for i, h in enumerate(all_hours)}
    T = len(all_hours)

    inflow_arr = np.zeros((T, N), dtype=np.float32)
    outflow_arr = np.zeros((T, N), dtype=np.float32)

    for _, row in df_pd.iterrows():
        t = hour_to_idx.get(row["hour"])
        if t is None:
            continue
        inflow_arr[t, int(row["dest_grid"])] += 1
        outflow_arr[t, int(row["origin_grid"])] += 1

    # 去极值 + log
    inflow_arr = np.log1p(inflow_arr)
    outflow_arr = np.log1p(outflow_arr)

    # 保存
    np.save(f"{prefix}_inflow.npy", inflow_arr)
    np.save(f"{prefix}_outflow.npy", outflow_arr)
    print(f"✅ {prefix} done: inflow {inflow_arr.shape}, outflow {outflow_arr.shape}\n")

    return inflow_arr, outflow_arr

# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    bike_in, bike_out = build_flow_numpy_polars(file_paths["bike"], columns_map["bike"], "bike")
    taxi_in, taxi_out = build_flow_numpy_polars(file_paths["taxi"], columns_map["taxi"], "taxi")
    print("🎉 所有 npy 文件生成完成！")