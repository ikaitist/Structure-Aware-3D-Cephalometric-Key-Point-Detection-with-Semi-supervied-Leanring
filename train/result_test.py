# 测试模型性能
import torch
from tests.coordinate_to_ras import data_loader
import pandas as pd
import SimpleITK as sitk
import numpy as np
from dataset import lable_loader


def find_max_indices(tensor):
    max_indices_list = []

    for i in range(5):
        max_value = torch.max(tensor[:, i, :, :, :])
        max_indices = torch.nonzero(tensor[:, i, :, :, :] == max_value, as_tuple=False)
        max_indices_list.append(max_indices)
    max_indices_tensor = torch.cat(max_indices_list, dim=0)
    max_indices_tensor = max_indices_tensor[:, 1:]
    return max_indices_tensor


def distance_count_resize(file_path, model_path, data_path, device, image_scale_size):
    # device = torch.device("cuda:2")
    all_distances = []
    while file_path <= 48:
        # load tensor
        tensor = torch.load(f'{model_path}/predict_{file_path}.pth')

        landmark_predict = find_max_indices(tensor)
        # print('landmark_predict:', landmark_predict)  # 像素坐标值

        label_path = f'{data_path}/{file_path}/{file_path}.csv'
        image_path = f'{data_path}/{file_path}/{file_path}.nii'
        # 得到spacing
        landmark_target = lable_loader(image_path, label_path, 5)
        # landmark_predict 需要缩放回去
        image = sitk.ReadImage(image_path)
        image_size = image.GetSize()
        scale = [image_size[i] / image_scale_size[i] for i in range(3)]
        scale = torch.tensor(scale, device=device)
        landmark_predict = landmark_predict * torch.tensor(scale, dtype=torch.float32)
        landmark_target = torch.tensor(landmark_target,device=device)
        # spacing = image.GetSpacing()
        # space = torch.tensor(spacing)
        # corr = data_loader(image_path, landmark_predict)
        # abs_corr_predicted = torch.abs(corr)
        # # print(abs_corr_predicted)       真实坐标值
        #
        # # 读取目标值
        # landmark_target = pd.read_csv(label_path, header=None)
        # landmark_target = torch.tensor(landmark_target.values, dtype=torch.float32)
        # landmark_target = torch.abs(landmark_target).to(device)
        # 计算距离
        distances = torch.norm(landmark_predict - landmark_target, dim=1)
        all_distances.extend(distances.cpu().numpy())
        # print(distances)
        distance_strings = [f"{d.item():.2f}" for d in distances]
        print(file_path)
        print(distance_strings)
        file_path += 1

    # 转换为NumPy数组
    all_distances = np.array(all_distances)

    # 计算平均误差和标准差
    mean_error = np.mean(all_distances)
    std_deviation = np.std(all_distances)

    # 定义阈值
    thresholds = [2, 2.5, 3, 4, 8]

    # 计算不同阈值下的检出率
    detection_rates = {thresh: np.mean(all_distances <= thresh) for thresh in thresholds}

    # 打印总体性能指标
    print(f"平均误差: {mean_error:.2f}mm")
    print(f"标准差: {std_deviation:.2f}mm")
    for thresh, rate in detection_rates.items():
        print(f"{thresh}mm 检出率: {rate:.2%}")

def distance_count(file_path, model_path, data_path, device):
    # device = torch.device("cuda:2")
    all_distances = []
    while file_path <= 48:
        # load tensor
        tensor = torch.load(f'{model_path}/predict_{file_path}.pth')

        landmark_predict = find_max_indices(tensor)
        # print('landmark_predict:', landmark_predict)  # 像素坐标值

        label_path = f'{data_path}/{file_path}/{file_path}.csv'
        image_path = f'{data_path}/{file_path}/{file_path}.nii'
        # 得到spacing
        image = sitk.ReadImage(image_path)
        spacing = image.GetSpacing()
        space = torch.tensor(spacing)
        corr = data_loader(image_path, landmark_predict)
        abs_corr_predicted = torch.abs(corr)
        # print(abs_corr_predicted)       真实坐标值

        # 读取目标值
        landmark_target = pd.read_csv(label_path, header=None)
        landmark_target = torch.tensor(landmark_target.values, dtype=torch.float32)
        landmark_target = torch.abs(landmark_target).to(device)
        # 计算距离
        distances = torch.norm(abs_corr_predicted - landmark_target, dim=1)
        all_distances.extend(distances.cpu().numpy())
        # print(distances)
        distance_strings = [f"{d.item():.2f}" for d in distances]
        print(file_path)
        print(distance_strings)
        file_path += 1

    # 转换为NumPy数组
    all_distances = np.array(all_distances)

    # 计算平均误差和标准差
    mean_error = np.mean(all_distances)
    std_deviation = np.std(all_distances)

    # 定义阈值
    thresholds = [2, 2.5, 3, 4, 8]

    # 计算不同阈值下的检出率
    detection_rates = {thresh: np.mean(all_distances <= thresh) for thresh in thresholds}

    # 打印总体性能指标
    print(f"平均误差: {mean_error:.2f}mm")
    print(f"标准差: {std_deviation:.2f}mm")
    for thresh, rate in detection_rates.items():
        print(f"{thresh}mm 检出率: {rate:.2%}")
