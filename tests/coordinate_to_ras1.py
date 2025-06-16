# 坐标转换：像素坐标得到RAS。
import torch
import SimpleITK as sitk
import numpy as np


def data_loader(coordinate):
    """读取.csv格式landmark,并将其RAS坐标系转换成世界坐标系对应体素"""
    image = 'tests/test.nii'
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

    # # load landmark
    # coordinate =
    # ras = torch.tensor(ras_pd.values, dtype=torch.float32)
    # # 根据公式转换
    coordinate = transform_coordinate(coordinate, origin, space, matrix)
    # print("coordinate=", coordinate, coordinate.dtype, coordinate.shape)
    return coordinate


def transform_coordinate(coordinate, origin, spacing, matrix):
    """转换公式"""
    # ras = ras - origin
    # # 转换矩阵 逆
    # matrix_inverse = torch.inverse(matrix)
    # # 执行矩阵乘法
    # ras = torch.matmul(matrix_inverse, ras.T).T
    # coordinate = torch.divide(ras, spacing)
    # coordinate_round = torch.as_tensor(coordinate, dtype=torch.int)
    spacing = spacing
    origin = origin
    matrix = matrix
    matrix_inverse = torch.inverse(matrix)

    ras = coordinate * spacing
    ras = torch.matmul(matrix_inverse, ras.T).T
    ras = ras + origin

    ras = torch.as_tensor(ras, dtype=torch.float32)
    return ras
