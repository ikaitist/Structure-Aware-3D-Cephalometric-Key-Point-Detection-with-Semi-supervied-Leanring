import os
import shutil
import re

# source_folder = '/home/user/imp/data/data_oh'  # 待分类文件夹路径
# train_folder = '/home/user/imp/data/data_gz/Train'  # 训练集文件夹路径
# test_folder = '/home/user/imp/data/data_gz/Test'  # 测试集文件夹路径
source_folder = '/DATA/lyh/data/data_oh'  # 待分类文件夹路径
train_folder = '/DATA/lyh/data/data_gz/Train'  # 训练集文件夹路径
test_folder = '/DATA/lyh/data/data_gz/Test'  # 测试集文件夹路径

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

def numeric_sort_key(filename):
    """ 提取文件名中的数字用于排序 """
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

# 遍历所有文件
files = os.listdir(source_folder)
for file in files:
    if file.endswith('.nii.gz') or file.endswith('.csv'):
        file_number = numeric_sort_key(file)[0]  # 获取文件名中的数字部分

        # 根据文件编号决定目标文件夹
        if 1 <= file_number <= 42:
            destination_folder = train_folder
        elif 43 <= file_number <= 48:
            destination_folder = test_folder
        else:
            continue  # 如果文件编号不在这个范围内，跳过

        # 构建目标文件夹路径
        folder_path = os.path.join(destination_folder, str(file_number))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 构建文件的完整路径
        source_file_path = os.path.join(source_folder, file)
        destination_file_path = os.path.join(folder_path, file)

        # 复制文件到对应的文件夹
        shutil.copy2(source_file_path, destination_file_path)
        print(f"Copied '{file}' to folder '{folder_path}'.")

print("Classification into train and test sets completed.")
