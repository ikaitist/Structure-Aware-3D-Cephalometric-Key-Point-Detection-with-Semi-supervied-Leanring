# 没用 测过了
import os
import shutil
import random

# 定义源数据文件夹路径和目标文件夹路径
source_dir = '/data/8T/lyh/data/data_K'
train_dir = os.path.join(source_dir, 'Train')
test_dir = os.path.join(source_dir, 'Test')

# 确保目标文件夹存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有文件列表
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# 设定比例并随机打乱文件列表
train_ratio = 5 / 6  # 5:1比例对应83.33%
random.shuffle(all_files)

# 划分文件列表
train_files = all_files[:int(len(all_files) * train_ratio)]
test_files = all_files[int(len(all_files) * train_ratio):]

# 移动文件到Train和Test文件夹
for file_name in train_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(train_dir, file_name))

for file_name in test_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(test_dir, file_name))

print("文件已按5:1的比例分配至Train和Test文件夹。")
