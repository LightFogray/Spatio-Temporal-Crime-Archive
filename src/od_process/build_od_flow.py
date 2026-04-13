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
    "bike": "./OD_data/divvy_clean_2022_2023.csv",
    "taxi": "./OD_data/taxi_clean_2022_2023.csv"
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
def time_to_bin_polars(df, time_col, new_col_name, prefix):
    """
    将时间字符串转换为小时级时间戳
    """
    print(f"  解析时间列: {time_col}")

    # 先查看原始时间格式样本
    sample_times = df.select(pl.col(time_col)).head(3).to_series().to_list()
    print(f"  时间样本: {sample_times}")
    # 已在过滤脚本中处理，目前自行车与出租车数据格式一致
    df = df.with_columns(
        pl.col(time_col)
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .dt.truncate("1h")
        .alias(new_col_name)
    )

    # 检查解析结果
    null_count = df.select(pl.col(new_col_name).is_null().sum()).to_series()[0]
    total_count = len(df)
    print(f"  时间解析: 成功 {total_count - null_count}/{total_count}, 失败 {null_count}")

    return df

# ==============================
# 构建 OD 流 (日级聚合版)
# ==============================
def build_flow_daily_polars(file_path, cols, prefix):
    """
    构建日级 OD 流特征，与 crime 数据对齐 (730 天)
    """
    print(f"📍 Processing {prefix} (daily aggregation)...")

    df = pl.read_csv(file_path, columns=list(cols))
    print(f"  原始记录数: {len(df):,}")

    # 丢掉空值
    df = df.drop_nulls()

    # 网格 ID
    df = coord_to_grid_polars(df, cols[0], cols[1], "origin_grid")
    df = coord_to_grid_polars(df, cols[2], cols[3], "dest_grid")

    # 检查坐标范围
    print(f"  坐标范围检查:")
    print(f"    出发经度: [{df[cols[0]].min():.4f}, {df[cols[0]].max():.4f}]")
    print(f"    出发纬度: [{df[cols[1]].min():.4f}, {df[cols[1]].max():.4f}]")
    print(f"    目的经度: [{df[cols[2]].min():.4f}, {df[cols[2]].max():.4f}]")
    print(f"    目的纬度: [{df[cols[3]].min():.4f}, {df[cols[3]].max():.4f}]")
    print(f"    网格系统: Lon [{LON_MIN}, {LON_MIN + NUM_X*GRID_SIZE}], Lat [{LAT_MIN}, {LAT_MIN + NUM_Y*GRID_SIZE}]")

    # 丢掉非法网格
    df_before = len(df)
    df = df.filter(
        (pl.col("origin_grid") >= 0) & (pl.col("origin_grid") < N) &
        (pl.col("dest_grid") >= 0) & (pl.col("dest_grid") < N)
    )
    print(f"  合法网格记录数: {len(df):,} / {df_before:,} ({len(df)/df_before*100:.1f}%)")

    # 时间解析
    df = time_to_bin_polars(df, cols[4], "hour", prefix)
    df = df.drop_nulls(subset=["hour"])

    # 检查是否还有有效数据
    if len(df) == 0:
        print(f"  错误: {prefix} 数据在时间解析后为空")
        return None, None

    # 添加日期列（用于日级聚合）
    df = df.with_columns(
        pl.col("hour").dt.truncate("1d").alias("date")
    )

    # 转 pandas
    df_pd = df.to_pandas()
    df_pd["hour"] = pd.to_datetime(df_pd["hour"], errors='coerce')
    df_pd["date"] = pd.to_datetime(df_pd["date"], errors='coerce')

    # 移除无效时间
    df_pd = df_pd.dropna(subset=["hour", "date"])

    min_date, max_date = df_pd["date"].min(), df_pd["date"].max()
    print(f"  日期范围: {min_date.date()} 至 {max_date.date()}")

    # 生成完整的日期范围 (730天)
    all_dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    T = len(all_dates)
    print(f"  期望时间步: {T} 天")

    # 日级聚合：按日期和网格统计
    inflow_daily = np.zeros((T, N), dtype=np.float32)
    outflow_daily = np.zeros((T, N), dtype=np.float32)

    # 使用 groupby 加速聚合
    grouped = df_pd.groupby(["date", "dest_grid"]).size().reset_index(name="inflow")
    print(f"  唯一目的网格数: {grouped['dest_grid'].nunique()}")
    for _, row in grouped.iterrows():
        t = date_to_idx.get(row["date"])
        if t is not None and 0 <= row["dest_grid"] < N:
            inflow_daily[t, int(row["dest_grid"])] = row["inflow"]

    grouped = df_pd.groupby(["date", "origin_grid"]).size().reset_index(name="outflow")
    print(f"  唯一出发网格数: {grouped['origin_grid'].nunique()}")
    for _, row in grouped.iterrows():
        t = date_to_idx.get(row["date"])
        if t is not None and 0 <= row["origin_grid"] < N:
            outflow_daily[t, int(row["origin_grid"])] = row["outflow"]

    # 诊断统计
    print(f"  inflow 非零元素: {(inflow_daily > 0).sum()}/{inflow_daily.size} ({(inflow_daily > 0).mean()*100:.2f}%)")
    print(f"  outflow 非零元素: {(outflow_daily > 0).sum()}/{outflow_daily.size} ({(outflow_daily > 0).mean()*100:.2f}%)")
    print(f"  inflow 最大值: {inflow_daily.max():.2f}")
    print(f"  outflow 最大值: {outflow_daily.max():.2f}")

    # 去极值 + log
    inflow_daily_log = np.log1p(inflow_daily)
    outflow_daily_log = np.log1p(outflow_daily)

    print(f"  log1p 后 inflow 范围: [{inflow_daily_log.min():.2f}, {inflow_daily_log.max():.2f}]")
    print(f"  log1p 后 outflow 范围: [{outflow_daily_log.min():.2f}, {outflow_daily_log.max():.2f}]")

    inflow_daily = inflow_daily_log
    outflow_daily = outflow_daily_log

    # 保存
    np.save(f"{prefix}_inflow_daily.npy", inflow_daily)
    np.save(f"{prefix}_outflow_daily.npy", outflow_daily)
    print(f"✅ {prefix} daily done: inflow {inflow_daily.shape}, outflow {outflow_daily.shape}\n")

    return inflow_daily, outflow_daily

# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    print("="*60)
    print("OD流特征构建（日级聚合）")
    print("="*60)

    # 生成日级 OD flow
    bike_in, bike_out = build_flow_daily_polars(file_paths["bike"], columns_map["bike"], "bike")
    taxi_in, taxi_out = build_flow_daily_polars(file_paths["taxi"], columns_map["taxi"], "taxi")

    # 检查是否有成功的结果
    success = True
    if bike_in is None:
        print("❌ bike 数据处理失败")
        success = False
    if taxi_in is None:
        print("❌ taxi 数据处理失败")
        success = False

    if success:
        print("\n" + "="*60)
        print("合并为 dynamic_od_flow_daily.npy...")
        print("="*60)

        # 合并为 (T, N, 4)
        T = bike_in.shape[0]
        N = bike_in.shape[1]

        # 对齐时间步（取最小）
        T_min = min(bike_in.shape[0], taxi_in.shape[0])
        if T_min < 730:
            print(f"  警告: 时间步 {T_min} < 730 天，数据可能不完整")

        od_flow = np.stack([
            bike_in[:T_min],
            bike_out[:T_min],
            taxi_in[:T_min],
            taxi_out[:T_min]
        ], axis=-1)

        print(f"  最终形状: {od_flow.shape} (T={T_min}, N={N}, channels=4)")
        print(f"  通道: 0=bike_in, 1=bike_out, 2=taxi_in, 3=taxi_out")

        np.save("dynamic_od_flow_daily.npy", od_flow.astype(np.float32))
        print(f"  保存到: dynamic_od_flow_daily.npy")

        print("\n" + "="*60)
        print("🎉 所有 npy 文件生成完成！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ 部分数据处理失败，请检查输入文件格式")
        print("="*60)