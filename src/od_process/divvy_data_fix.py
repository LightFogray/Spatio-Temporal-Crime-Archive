import os
import shutil

def move_csv_to_current():
    """
    将当前目录下所有子文件夹中的CSV文件移动到当前目录
    """
    current_dir = r"D:\mycode\crime-temporal-spatio\chicago_data\divvy"
    print(f"当前目录: {current_dir}")
    
    moved_count = 0
    
    # 遍历所有子文件夹
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(item_path):
            print(f"检查文件夹: {item}")
            
            # 查找文件夹中的CSV文件
            for file in os.listdir(item_path):
                if file.lower().endswith('.csv'):
                    source = os.path.join(item_path, file)
                    target = os.path.join(current_dir, file)
                    
                    # 处理重名
                    if os.path.exists(target):
                        name, ext = os.path.splitext(file)
                        new_name = f"{name}_{item}{ext}"
                        target = os.path.join(current_dir, new_name)
                        print(f"  文件已存在，重命名为: {new_name}")
                    
                    # 移动文件
                    shutil.move(source, target)
                    moved_count += 1
                    print(f"  已移动: {file}")
    
    print(f"\n完成！共移动 {moved_count} 个CSV文件到当前目录")

if __name__ == "__main__":
    move_csv_to_current()