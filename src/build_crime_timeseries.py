import pandas as pd
import geopandas as gpd
import numpy as np
import os

def build_timeseries(crime_csv, grid_shp, output_npy, crime_type_name="crime"):
    """
    构建犯罪时间序列矩阵

    Args:
        crime_csv: 犯罪数据CSV路径
        grid_shp: 网格shapefile路径
        output_npy: 输出npy文件路径
        crime_type_name: 犯罪类型名称（用于日志）

    Returns:
        crime_matrix: 时间序列矩阵 (time_steps, n_grids)
        date_range: 日期范围
    """
    print(f"\n[{crime_type_name}] Loading data from {crime_csv}")

    if not os.path.exists(crime_csv):
        print(f"  Error: File not found: {crime_csv}")
        return None, None

    crime = pd.read_csv(crime_csv)
    print(f"  Loaded {len(crime)} records")

    grid = gpd.read_file(grid_shp)
    print(f"  Grid cells: {len(grid)}")

    # 时间处理
    crime["date"] = pd.to_datetime(crime["date"], errors="coerce")
    crime = crime.dropna(subset=["date"])
    crime["date"] = crime["date"].dt.floor("D")

    # 删除无坐标
    crime = crime.dropna(subset=["latitude", "longitude"])
    print(f"  Records with valid coords: {len(crime)}")

    # 转geodataframe
    crime_gdf = gpd.GeoDataFrame(
        crime,
        geometry=gpd.points_from_xy(crime["longitude"], crime["latitude"]),
        crs="EPSG:4326"
    )

    crime_gdf = crime_gdf.to_crs(grid.crs)

    # 空间匹配
    crime_join = gpd.sjoin(
        crime_gdf,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within"
    )

    # 统计每日每网格犯罪数
    crime_matrix = crime_join.groupby(["date", "grid_id"]).size().unstack(fill_value=0)

    # 所有grid id
    all_grids = grid["grid_id"].values

    # 时间补齐
    date_range = pd.date_range(
        start=crime_matrix.index.min(),
        end=crime_matrix.index.max(),
        freq="D"
    )

    crime_matrix = crime_matrix.reindex(
        index=date_range,
        columns=all_grids,
        fill_value=0
    )

    # 转numpy
    crime_array = crime_matrix.values.astype(np.float32)

    print(f"  Crime matrix shape: {crime_array.shape} (time_steps={crime_array.shape[0]}, grids={crime_array.shape[1]})")
    print(f"  Date range: {date_range[0].date()} to {date_range[-1].date()}")
    print(f"  Daily mean: {crime_array.mean():.4f}")
    print(f"  Daily max: {crime_array.max():.4f}")

    # 保存
    np.save(output_npy, crime_array)
    print(f"  Saved to: {output_npy}")

    return crime_array, date_range


def main():
    """主函数：分别构建暴力和财产犯罪时间序列"""

    grid_shp = "data/processed/chicago_grid.shp"
    cleaned_dir = "chicago_crime_data/cleaned"

    # 确保输出目录存在
    os.makedirs("data/processed", exist_ok=True)

    all_results = {}
    date_ranges = {}

    # 1. 处理暴力犯罪
    violent_files = [
        (f"{cleaned_dir}/violent_2022_cleaned.csv", 2022),
        (f"{cleaned_dir}/violent_2023_cleaned.csv", 2023),
    ]

    violent_dfs = []
    for f, year in violent_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df["year"] = year
            violent_dfs.append(df)
            print(f"Loaded violent {year}: {len(df)} records")

    if violent_dfs:
        violent_all = pd.concat(violent_dfs, ignore_index=True)
        temp_csv = "data/processed/temp_violent_all.csv"
        violent_all.to_csv(temp_csv, index=False)

        violent_array, violent_dates = build_timeseries(
            temp_csv,
            grid_shp,
            "data/processed/crime_violent_timeseries.npy",
            "Violent Crime"
        )
        all_results["violent"] = violent_array
        date_ranges["violent"] = violent_dates

        os.remove(temp_csv)

    # 2. 处理财产犯罪
    property_files = [
        (f"{cleaned_dir}/property_2022_cleaned.csv", 2022),
        (f"{cleaned_dir}/property_2023_cleaned.csv", 2023),
    ]

    property_dfs = []
    for f, year in property_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df["year"] = year
            property_dfs.append(df)
            print(f"Loaded property {year}: {len(df)} records")

    if property_dfs:
        property_all = pd.concat(property_dfs, ignore_index=True)
        temp_csv = "data/processed/temp_property_all.csv"
        property_all.to_csv(temp_csv, index=False)

        property_array, property_dates = build_timeseries(
            temp_csv,
            grid_shp,
            "data/processed/crime_property_timeseries.npy",
            "Property Crime"
        )
        all_results["property"] = property_array
        date_ranges["property"] = property_dates

        os.remove(temp_csv)

    # 3. 合并两类犯罪（可选）
    if "violent" in all_results and "property" in all_results:
        print("\n[Merging] Creating combined crime timeseries...")

        # 确保时间范围一致
        start_date = max(dr[0] for dr in date_ranges.values())
        end_date = min(dr[-1] for dr in date_ranges.values())

        # 截取相同时间范围
        violent_slice = all_results["violent"][
            (date_ranges["violent"] >= start_date) &
            (date_ranges["violent"] <= end_date)
        ]
        property_slice = all_results["property"][
            (date_ranges["property"] >= start_date) &
            (date_ranges["property"] <= end_date)
        ]

        # 合并：可以作为两个通道 (time, grids, 2)
        combined_array = np.stack([violent_slice, property_slice], axis=-1)

        print(f"  Combined shape: {combined_array.shape} (time, grids, channels)")
        np.save("data/processed/crime_combined_timeseries.npy", combined_array)
        print(f"  Saved to: data/processed/crime_combined_timeseries.npy")

        # # 或者简单相加作为总量 (time, grids)
        # total_array = violent_slice + property_slice
        # np.save("data/processed/crime_total_timeseries.npy", total_array)
        # print(f"  Total crime shape: {total_array.shape}")
        # print(f"  Saved to: data/processed/crime_total_timeseries.npy")

    print("\nAll timeseries generated successfully!")


if __name__ == "__main__":
    main()
