import numpy as np

files = [
# "poi_features.npy",
# "road_density.npy",
# "landuse_features.npy",
# "green_ratio.npy",
# "nightlight_features.npy",
# "weather_features.npy",
"crime_grid_timeseries.npy",
# "X.npy",
# "Y.npy",
# "adj_adaptive.npy",
# "adj_distance.npy",
# "adj_crime_dynamic_gaussian.npy",
"bike_inflow_1246.npy",
"bike_outflow_1246.npy",
"taxi_inflow_1246.npy",
"taxi_outflow_1246.npy",
"dynamic_od_flow_1246.npy",
"adj_crime_dynamic_od.npy"
]

for f in files:

    data = np.load("data/processed/"+f,allow_pickle=True)

    print("------")
    print(f)
    print("shape:",data.shape)
    print("dtype:",data.dtype)