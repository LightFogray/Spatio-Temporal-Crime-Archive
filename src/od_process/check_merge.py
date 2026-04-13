import pandas as pd
import glob
import os

def check_csv_files(folder_path):
    """
    检查CSV文件的行数（排除空行），并验证合并后的预期行数
    
    参数:
        folder_path: CSV文件所在的文件夹路径
    """
    
    # 获取所有CSV文件
    all_csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    csv_files = [f for f in all_csv_files if not os.path.basename(f).lower().startswith('combined')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print("=" * 60)
    print("CSV文件行数检查报告（已排除空行）")
    print("=" * 60)
    
    # 存储每个文件的有效行数（排除空行）
    file_row_counts = {}
    file_raw_counts = {}
    total_expected_rows = 0
    
    for i, file in enumerate(csv_files):
        filename = os.path.basename(file)
        
        try:
            # 方法1：使用pandas读取，自动排除全空行
            df = pd.read_csv(file)
            
            # 方法2：手动计算行数，排除所有值都为空的行
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 原始行数（包括空行）
                raw_line_count = len(lines)
                
                # 排除空行（strip后为空的行）
                non_empty_lines = [line for line in lines if line.strip()]
                
                # 排除空行后的行数
                cleaned_line_count = len(non_empty_lines)
                
                # 对于CSV，表头也算一行，所以总行数需要处理
                if i == 0:  # 第一个文件，表头保留
                    valid_rows = cleaned_line_count  # 包含表头
                    expected_contribution = cleaned_line_count  # 全部计入
                else:  # 其他文件，表头需要排除
                    valid_rows = cleaned_line_count  # 包含表头
                    expected_contribution = cleaned_line_count - 1 if cleaned_line_count > 0 else 0
            
            file_raw_counts[filename] = raw_line_count
            file_row_counts[filename] = valid_rows
            total_expected_rows += expected_contribution
            
            # 输出每个文件的详细信息
            print(f"\n文件 {i+1}: {filename}")
            print(f"  ├─ 原始行数（含空行）: {raw_line_count}")
            print(f"  ├─ 有效行数（排除空行后）: {valid_rows}")
            print(f"  ├─ 其中表头行: {'是（将保留）' if i == 0 else '是（将排除）'}")
            print(f"  └─ 对合并后行数的贡献: {expected_contribution} 行")
            
        except Exception as e:
            print(f"\n文件 {filename} 读取失败: {e}")
    
    # 输出汇总信息
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    print("\n各文件原始行数详情（原始文件中的行数，含空行）:")
    for i, (file, count) in enumerate(file_raw_counts.items()):
        print(f"  {i+1}. {file}: {count} 行")
    
    print(f"\n原始文件总行数（含空行，含重复表头）: {sum(file_raw_counts.values())}")
    print(f"合并后预期行数（仅第一个文件保留表头，排除空行）: {total_expected_rows}")
    
    # 检查合并后的文件是否存在
    combined_file = os.path.join(folder_path, "combined_taxi_trips_data_20210101-20250101.csv")
    if os.path.exists(combined_file):
        try:
            combined_df = pd.read_csv(combined_file)
            actual_rows = len(combined_df)
            print(f"\n合并后实际文件行数（排除空行）: {actual_rows}")
            
            if actual_rows == total_expected_rows:
                print("\n✅ 检查通过！实际行数与预期行数一致。")
            else:
                print(f"\n❌ 警告：行数不匹配！")
                print(f"   预期: {total_expected_rows} 行，实际: {actual_rows} 行")
                print(f"   差异: {actual_rows - total_expected_rows} 行")
        except Exception as e:
            print(f"\n无法读取合并后的文件: {e}")
    else:
        print("\n⚠️  合并后的文件 'combined_bus_data.csv' 不存在")
    
    return file_row_counts, total_expected_rows

# 使用示例
if __name__ == "__main__":
    # 设置文件夹路径
    folder_path = r"D:\\crime\\OD_data\\taxi\\"  # 修改为你的文件夹路径
    
    # 运行检查
    check_csv_files(folder_path)

