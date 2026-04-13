import numpy as np

files = [
# "poi_features.npy",
# "road_density.npy",
# "landuse_features.npy",
# "green_ratio.npy",
# "nightlight_features.npy",
"weather_features.npy",
"crime_combined_timeseries.npy",
# "X.npy",
# "Y.npy",
# "adj_adaptive.npy",
# "adj_distance.npy",
# "adj_crime_dynamic_gaussian.npy",
"bike_inflow_1246.npy",
"bike_outflow_1246.npy",
"taxi_inflow_1246.npy",
"taxi_outflow_1246.npy",
"dynamic_od_flow.npy",
# "adj_crime_dynamic_od.npy",
# "semantic_embedding.npy"
]

for f in files:

    data = np.load("data/processed/"+f,allow_pickle=True)

    print("------")
    print(f)
    print("shape:",data.shape)
    print("dtype:",data.dtype)
    print("min:", data.min())
    print("max:", data.max())
# import numpy as np

# green = np.load("data/processed/green_features.npy")
# nan_indices = np.where(np.isnan(green))[0]

# print(f"NaN网格索引: {nan_indices}")
# print(f"NaN比例: {len(nan_indices)/len(green)*100:.1f}%")

# # 查看这些网格的其他特征
# poi = np.load("data/processed/poi_features.npy")
# landuse = np.load("data/processed/landuse_features.npy")

# for idx in nan_indices[:5]:  # 看前5个
#     print(f"\n网格 {idx}:")
#     print(f"  POI: {poi[idx]}")
#     print(f"  土地利用: {landuse[idx]}")
