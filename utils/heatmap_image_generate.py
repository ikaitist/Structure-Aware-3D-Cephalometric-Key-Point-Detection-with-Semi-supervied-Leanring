import torch
import numpy as np


def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False):
    ### 有batchsize
    landmarks_shape = landmarks.shape
    sigmas_shape = sigmas.shape

    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2]
    assert len(heatmap_size) == dim, 'Dimensions do not match.'
    assert sigmas_shape[0] == num_landmarks, 'Number of sigmas does not match.'

    #  landmarks[..., 1:] 所有行和从第二列开始的所有列
    heatmap_axis = 1
    # 删除了有效位，做出了一定更改
    landmarks_reshaped = torch.reshape(landmarks, [batch_size, num_landmarks] + [1] * dim + [dim])
    # is_valid_reshaped = torch.reshape(landmarks[..., 0], [batch_size, num_landmarks] + [1] * dim)
    sigmas_reshaped = torch.reshape(sigmas, [1, num_landmarks] + [1] * dim)
    print(landmarks_reshaped.shape)
    print(sigmas_reshaped.shape)

    aranges = [torch.tensor(np.arange(s)) for s in heatmap_size]
    # grid = tf.meshgrid(*aranges, indexing='ij')

    # 使用 torch.meshgrid() 创建多维网格（ij索引）
    grid = torch.meshgrid(*aranges, indexing='ij')

    grid_stacked = torch.stack(grid, dim=dim)
    grid_stacked = grid_stacked.type(torch.FloatTensor)
    # grid_stacked = grid_stacked.cuda()
    # 在维度0上将 grid_stacked 复制 batch_size 次并堆叠
    grid_stacked = torch.stack([grid_stacked] * batch_size, dim=0)
    # 在 heatmap_axis 维度上将 grid_stacked 复制 num_landmarks 次并堆叠
    grid_stacked = torch.stack([grid_stacked] * num_landmarks, dim=heatmap_axis)

    if normalize:
        scale /= torch.pow(np.sqrt(2 * np.pi) * sigmas_reshaped, dim)
    scale = torch.tensor(scale)

    squared_distances = torch.sum(torch.pow(grid_stacked - landmarks_reshaped, 2.0), dim=-1)
    heatmap = scale * torch.exp(-squared_distances / (2 * torch.pow(sigmas_reshaped, 2)))
    print("heatmap:", heatmap.shape)
    # heatmap_or_zeros = torch.where((is_valid_reshaped + torch.zeros_like(heatmap)) > 0, heatmap,torch.zeros_like(heatmap))
    heatmap_or_zeros = torch.where(heatmap > 0, heatmap, torch.zeros_like(heatmap))
    # heatmap_or_zeros = heatmap_or_zeros.transpose(2, 4)

    return heatmap_or_zeros


def generate_heatmap_target1(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False):
    ### 没有batchsize
    landmarks_shape = landmarks.shape
    sigmas_shape = sigmas.shape
    # batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[0]
    dim = landmarks_shape[1]

    assert len(heatmap_size) == dim, 'Dimensions do not match.'
    assert sigmas_shape[0] == num_landmarks, 'Number of sigmas does not match.'
    #  landmarks[..., 1:] 所有行和从第二列开始的所有列
    heatmap_axis = 0
    # 删除了有效位，做出了一定更改
    landmarks_reshaped = torch.reshape(landmarks, [num_landmarks] + [1] * dim + [dim])
    # is_valid_reshaped = torch.reshape(landmarks[..., 0], [batch_size, num_landmarks] + [1] * dim)
    sigmas_reshaped = torch.reshape(sigmas, [num_landmarks] + [1] * dim)
    aranges = [torch.tensor(np.arange(s)) for s in heatmap_size]
    # grid = tf.meshgrid(*aranges, indexing='ij')
    # 使用 torch.meshgrid() 创建多维网格（ij索引）
    grid = torch.meshgrid(*aranges, indexing='ij')
    grid_stacked = torch.stack(grid, dim=dim)
    grid_stacked = grid_stacked.type(torch.FloatTensor)
    # grid_stacked = grid_stacked.cuda()
    # 在维度0上将 grid_stacked 复制 batch_size 次并堆叠
    # batch_size =1
    # grid_stacked = torch.stack([grid_stacked] * batch_size, dim=0)
    # 在 heatmap_axis 维度上将 grid_stacked 复制 num_landmarks 次并堆叠
    grid_stacked = torch.stack([grid_stacked] * num_landmarks, dim=heatmap_axis)

    if normalize:
        scale /= torch.pow(np.sqrt(2 * np.pi) * sigmas_reshaped, dim)
    scale = torch.as_tensor(scale)
    # print("sigmas_reshaped:", sigmas_reshaped.shape)
    # print("scale:", scale.shape)
    # print("grid_stacked:", grid_stacked.shape)
    # print("landmarks_reshaped:", landmarks_reshaped.shape)
    squared_distances = torch.sum(torch.pow(grid_stacked - landmarks_reshaped, 2.0), dim=-1)
    # print("squared_distances:", squared_distances.shape)

    heatmap = scale * torch.exp(-squared_distances / (2 * torch.pow(sigmas_reshaped, 2)))
    # print("heatmap:", heatmap.shape)
    # exit()
    # heatmap_or_zeros = torch.where((is_valid_reshaped + torch.zeros_like(heatmap)) > 0, heatmap,torch.zeros_like(heatmap))
    heatmap_or_zeros = torch.where(heatmap > 0, heatmap, torch.zeros_like(heatmap))
    # heatmap_or_zeros = heatmap_or_zeros.transpose(2, 4)

    return heatmap_or_zeros

def generate_heatmap_target2(heatmap_size, landmarks, x):
    num_landmarks = landmarks.shape[0]
    dim = landmarks.shape[1]

    assert len(heatmap_size) == dim, 'Dimensions do not match.'

    landmarks_reshaped = torch.reshape(landmarks, [num_landmarks] + [1] * dim + [dim])

    aranges = [torch.tensor(np.arange(s)) for s in heatmap_size]
    grid = torch.meshgrid(*aranges, indexing='ij')
    grid_stacked = torch.stack(grid, dim=dim).type(torch.FloatTensor)

    grid_stacked = torch.stack([grid_stacked] * num_landmarks, dim=0)

    # 计算每个点与地标的平方距离
    squared_distances = torch.sum(torch.pow(grid_stacked - landmarks_reshaped, 2.0), dim=-1)

    # 应用阈值，小于等于 x*x 的地方设为 1，其他地方设为 0
    heatmap = torch.where(squared_distances <= x*x, torch.ones_like(squared_distances), torch.zeros_like(squared_distances))
    indices_of_ones = torch.nonzero(heatmap, as_tuple=False)
    return heatmap