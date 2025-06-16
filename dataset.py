import torch
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import re
import json


class DataSet(object):
    def __init__(self,
                 images,
                 labels, ):
        # assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
        self.images = images
        self.labels = labels


# def get_file_list(file_dir):
#     image_list = []
#     for d in os.listdir(file_dir):
#         path = '{}/{}'.format(file_dir, d)
#         image_list.append(path)
#     return image_list

def numeric_sort_key(filename):
    """ 提取文件名中的数字用于排序 """
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]


def get_file_list_gz(file_dir):
    image_list = []
    label_list = []
    for dir in sorted(os.listdir(file_dir), key=numeric_sort_key):
        child_dir = os.path.join(file_dir, dir)
        for d in sorted(os.listdir(child_dir), key=numeric_sort_key):
            path = os.path.join(child_dir, d)
            # 检查文件扩展名是否为 .nii.gz
            if d.endswith('.nii.gz'):
                image_list.append(path)
            else:
                label_list.append(path)
    return image_list, label_list


def get_file_list(file_dir):
    image_list = []
    label_list = []
    for dir in sorted(os.listdir(file_dir), key=numeric_sort_key):
        child_dir = os.path.join(file_dir, dir)
        for d in sorted(os.listdir(child_dir), key=numeric_sort_key):
            path = os.path.join(child_dir, d)
            if os.path.splitext(d)[1] in ['.nii']:
                image_list.append(path)
            else:
                label_list.append(path)
    return image_list, label_list


def get_file_list_unlabeled(file_dir, extension):
    file_list = []
    # 遍历目录
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(extension):
                # 拼接完整的文件路径
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    # 按文件名中的数字排序
    file_list.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))

    return file_list


def JsonLoader(label_path):
    with open(label_path, 'r') as file:
        # 加载JSON数据并解析到一个字典中
        data = json.load(file)
        control_points = data["markups"][0]["controlPoints"]
        # 创建一个空列表来存储所有点的坐标
        points = []
        # 遍历控制点并提取坐标
        for point in control_points:
            position = point["position"]
            points.append(position)

        # 将点的列表转换为PyTorch张量
        tensor_points = torch.tensor(points, dtype=torch.int32)
        return tensor_points


def lable_loader(image, label, num_landmark):
    """读取.csv格式landmark,并将其RAS坐标系转换成世界坐标系对应体素"""
    image = sitk.ReadImage(image)
    # get image spacing
    spacing = image.GetSpacing()
    space = torch.tensor(spacing)
    # print("space", space, space.dtype)
    # get image origin
    ori = image.GetOrigin()
    origin = torch.tensor(ori)

    # change origin
    indices = [0, 1]
    origin[indices] = -origin[indices]
    # print("origin=", origin, origin.dtype)

    # transform matrix
    matrix = torch.FloatTensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    # print("matrix=", matrix, matrix.dtype)

    # load landmark
    ras_pd = pd.read_csv(label, header=None)
    ras = torch.tensor(ras_pd.values, dtype=torch.float32)
    # 根据公式转换
    coordinate = transform_coordinate(ras, origin, space, matrix)
    # print("coordinate=", coordinate, coordinate.dtype, coordinate.shape)
    return coordinate


def transform_coordinate(ras, origin, spacing, matrix):
    """转换公式"""
    ras = ras - origin
    # 转换矩阵 逆
    matrix_inverse = torch.inverse(matrix)
    # 执行矩阵乘法
    ras = torch.matmul(matrix_inverse, ras.T).T
    coordinate = torch.divide(ras, spacing)
    coordinate_round = torch.as_tensor(coordinate, dtype=torch.int)
    coordinate_round = torch.abs(coordinate_round)
    return coordinate_round
