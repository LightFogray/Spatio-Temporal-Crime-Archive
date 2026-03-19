import pandas as pd
import glob
import os

folder_path = r"D:\\crime\\OD_data\\divvy\\"
csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
output_file = os.path.join(folder_path, "combined_divvy_trips_data_20210101-20250101.csv")

if not csv_files:
    print("未找到CSV文件")
else:
    # 处理第一个文件（写入表头）
    first_df = pd.read_csv(csv_files[0])
    first_df.to_csv(output_file, index=False)
    print(f"已写入第一个文件: {os.path.basename(csv_files[0])}")
    
    # 追加其余文件（不写表头）
    for file in csv_files[1:]:
        # 分块读取，避免内存问题
        for chunk in pd.read_csv(file, skiprows=1, header=None, 
                                names=first_df.columns, chunksize=10000):
            chunk.to_csv(output_file, mode='a', header=False, index=False)
        print(f"已追加文件: {os.path.basename(file)}")
    
    print(f"所有文件合并完成！")