from src.download_data import download_chicago_data,download_road_network,download_landuse,download_greenland
from src.build_grid import build_grid
from build_static_features import build_poi_features,build_road_features,build_landuse_features,build_green_features
from src.build_graph import build_adj_matrix


if __name__ == "__main__":

    # 1 下载数据（只执行一次）
    # download_chicago_data()
    # download_landuse()
    # download_greenland()
    # download_road_network()


    # 2 构建网格
    # build_grid()

    # 3 构建POI特征
    # build_poi_features()
    # build_landuse_features()
    # build_green_features()
    build_road_features()

    # 4 构建邻接矩阵
    # build_adj_matrix()