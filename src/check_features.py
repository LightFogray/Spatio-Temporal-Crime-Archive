import numpy as np

files = [
# "poi_features.npy",
# "road_density.npy",
# "landuse_features.npy",
# "green_ratio.npy",
# "nightlight_features.npy",
# "weather_features.npy",
# "crime_grid_timeseries.npy",
# "mixed_graph_adj_matrix.npy",
"X.npy",
"Y.npy",
"adj_adaptive.npy",
"adj_distance.npy",
"adj_crime_dynamic_gaussian.npy",
# "X_val.npy",
# "Y_val.npy",
# "X_test.npy",
# "Y_test.npy",
]

for f in files:

    data = np.load("data/processed/"+f,allow_pickle=True)

    print("------")
    print(f)
    print("shape:",data.shape)
    print("dtype:",data.dtype)