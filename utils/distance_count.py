import torch
from tests.coordinate_to_ras1 import data_loader


def find_max_indices(tensor):
    max_indices_list = []
    for i in range(5):
        max_value = torch.max(tensor[:, i, :, :, :])
        max_indices = torch.nonzero(tensor[:, i, :, :, :] == max_value, as_tuple=False)
        max_indices_list.append(max_indices)
    max_indices_tensor = torch.cat(max_indices_list, dim=0)
    max_indices_tensor = max_indices_tensor[:, 1:]
    return max_indices_tensor


def count_distances(image_ori, image_pre):
    # 找到预测图中最大值
    max_indices_pre = find_max_indices(image_pre)
    corr_pre = data_loader(max_indices_pre)
    abs_corr_pre = torch.abs(corr_pre)
    # print(abs_corr_pre)       真实坐标值
    # 读取真实值
    max_indices_ori = find_max_indices(image_ori)
    corr_ori = data_loader(max_indices_ori)
    abs_corr_ori = torch.abs(corr_ori)

    # 计算距离
    distances = torch.norm(abs_corr_pre - corr_ori, dim=1)
    distance_strings = [f"{d.item():.2f}" for d in distances]
    return distance_strings
