import random
import torch
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
import json
from dataset import get_file_list_gz, get_file_list_unlabeled
from dataset import lable_loader
from torch.utils.data import Dataset, DataLoader
from model import Unet3d, fine_gat
# from tests.model_scn import Unet3d
from tqdm import tqdm
from torch.optim import lr_scheduler
# from monai.networks.nets import SwinUNETR
from network_layer.swin_unetr import SwinUNETR
from utils.heatmap_image_generate import generate_heatmap_target1
from utils.unravel_index import unravel_index
from tensorboardX import SummaryWriter
from unter import UNETR
from datetime import datetime
import torch.nn.functional as F
import os
from torch.cuda.amp import GradScaler, autocast
from utils.Utils import loss_function
from scheduler.scheduler import LinearWarmupCosineAnnealingLR

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# device = torch.device("cuda:0")

# 设置随机种子

param = 'SwinUnetr-gat_E_200_L=0.0001_S=128_factor=0.95_cropSize=24_alpha=0.4_prop=10000_con'
# param = 'test'

pretrain_model_param = "Baseline_SwinUnetr_E_300_L=0.0015_S=128_factor=0.95_2.25"
print(param)
# baselin=>'SwinUnetr_lr=1e-4_size96_seg_ReduceLROnPlateau_2_batchsize_1'
# 创建用于存储日志的文件夹
cache = './cache'
logs_folder = cache + '/logs/' + param
os.makedirs(logs_folder, exist_ok=True)
writer = SummaryWriter(logs_folder)
# 创建用于存储模型的文件夹
model_saved = cache + '/model_saved/' + param + '/'
print(model_saved)
os.makedirs(model_saved, exist_ok=True)
available_devices = [1]  # 指定可用的GPU设备编号
data = './data'


class Config(object):
    # File paths
    data_train_dir = data + "/data_k/Train"
    data_test_dir = data + "/data_k/Test"
    trainNet = "SwinUNETR"
    feature_size = 48  # 12的倍数
    # trainNet = "Unet"
    # lable_dir = 'data'
    image_list = []
    label_list = []
    epoch = 201
    batch_size = 1
    image_size = [128, 128, 128]
    heatmap_size = image_size
    heatmap_scale = 1000
    num_landmark = 24
    heatmap_sigma = torch.nn.Parameter(torch.full((num_landmark,), 4.0))
    leaning_rate = 0.0001
    alpha = 0.4
    prop = 10000
    crop_size = 24
    device = torch.device('cuda', available_devices[0])

class AdaptiveKeypointUncertaintyLoss(nn.Module):
    """
    自适应关键点回归损失（3D 坐标版本），基于同方差不确定性学习。
    输入 preds 和 targets 形状应为 (B, N, 3)，其中
      - B: 批量大小
      - N: 关键点数量
      - 3: XYZ 坐标维度

    对于第 i 个关键点：
        L_i = 1/(2σ_i^2) * ||y_i - ŷ_i||^2 + log σ_i
    最终损失 = 对所有关键点和样本的加权累加或平均。

    用法示例：
        # 在训练脚本初始化，指定 device 参数
        loss_fn = AdaptiveKeypointUncertaintyLoss(
            num_keypoints=config.num_landmark,
            reduction='mean',
            device=config.device
        )

        # 训练过程中调用
        fine_loss = loss_fn(fine_predict, coordinate_scaled)
    """
    def __init__(self, num_keypoints: int, reduction: str = 'mean', device: torch.device = torch.device('cpu'), eps: float = 1e-6):
        super().__init__()
        assert isinstance(num_keypoints, int) and num_keypoints > 0, \
            f"num_keypoints must be int > 0, got {num_keypoints}"
        assert reduction in ('mean', 'sum'), "reduction must be 'mean' or 'sum'"

        self.device = device
        self.reduction = reduction
        self.eps = eps
        # 在指定 device 上初始化 log_sigma
        self.log_sigma = nn.Parameter(
            torch.zeros(num_keypoints, device=device), requires_grad=True
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds/targets: (B, N, 3)
        assert preds.dim() == 3 and preds.size(2) == 3, \
            f"Expected preds/targets shape (B, N, 3), got {preds.shape}"
        assert preds.shape == targets.shape, \
            f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"

        B, N, _ = preds.shape
        # 计算每关键点的欧氏平方误差 (B, N)
        squared_error = torch.sum((preds - targets) ** 2, dim=-1)

        # log_sigma 已在正确 device 上，直接扩展
        log_sigma = self.log_sigma.unsqueeze(0).expand(B, N)
        inv_sigma_sq = torch.exp(-2 * log_sigma)

        # 按式 L_i = 0.5 * inv_sigma_sq * se + log_sigma
        loss_per_point = 0.5 * inv_sigma_sq * squared_error + log_sigma
        # 对关键点求和得到每个样本的损失 (B,)
        loss_per_sample = loss_per_point.sum(dim=1)

        # 批量 reduction
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        return loss_per_sample.sum()

class Datasets_loader(Dataset):
    def __init__(self, image_list, label_list, transformer=None):
        """
        通用数据集类，适用于训练集和测试集。

        Args:
            image_list (list): 图像路径列表
            label_list (list): 标签路径列表
            transformer (callable, optional): 图像变换函数，默认为 None
        """
        self.label_list = label_list
        self.image_list = image_list
        self.transformer = transformer  # 统一为一个可选参数
        self.config = Config()

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]

        # 加载图像和标签
        image, label, filename = process_data(image_path, label_path, self.config)

        # 如果 transformer 存在，则对图像进行变换
        if self.transformer:
            image = self.transformer(image)

        return image, label, filename

    def __len__(self):
        return len(self.image_list)


def main():
    config = Config()
    device = config.device

    image_train_list, label_train_list = get_file_list_gz(config.data_train_dir)  # done get file list
    image_test_list, label_test_list = get_file_list_gz(config.data_test_dir)
    transformer = 0  # 待加入
    # 加载训练集，测试集
    datasets_train = Datasets_loader(image_train_list, label_train_list, transformer)
    datasets_test = Datasets_loader(image_test_list, label_test_list, transformer)

    datasets_train = DataLoader(datasets_train, batch_size=config.batch_size, shuffle=False)
    datasets_test = DataLoader(datasets_test, batch_size=config.batch_size, shuffle=True)

    # model = Unet3d(in_channels=1, n_classes=5, n_channels=32).to(device)  # 网络
    # model = UNETR(
    #     in_channels=1,
    #     out_channels=5,
    #     img_size=(128, 128, 128),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.0).to(device)
    if (config.trainNet == "SwinUNETR"):
        model = SwinUNETR(
            in_channels=1,
            out_channels=config.num_landmark,
            img_size=(128, 128, 128),
            feature_size=48, ).to(device)
    else:
        model = Unet3d(in_channels=1, n_classes=config.num_landmark, n_channels=32).to(device)  # 网络

    gat_model = fine_gat(config).to(device)
    # 加载预训练模型
    # model.load_state_dict(torch.load(pretrained_model_path, map_location=lambda storage, loc: storage.cuda(2)))
    # pretrained_model_path = cache + '/model_saved/Baseline_SwinUnetr_E_300_L=0.0015_S=128_factor=0.95_2.25/checkpoint_epoch_5.pt'
    # checkpoint = torch.load(pretrained_model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # gat_model.load_state_dict(checkpoint['gat_model_state_dict'])
    loss_mse = loss_function  # 损失函数
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gat_model.parameters()),  # 将两个模型的参数传入优化器
        lr=config.leaning_rate
    )  # 优化器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=10)

    # scheduler = LinearWarmupCosineAnnealingLR(
    #         optimizer, warmup_epochs=30, max_epochs=config.epoch)

    # 添加学习率调度器，例如使用StepLR在每20个epoch减少学习率到原来的0.1倍
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    # model = model.to(device)
    scaler = GradScaler()
    print(
        "------------------------------------------------start training--------------------------------------------")
    for epoch in tqdm(range(config.epoch)):  # 训练

        loss, distance = (
            train_and_eval(model, gat_model, datasets_train, datasets_test, optimizer, loss_mse, device, scaler,
                           config))  # device: gpu
        train_loss = loss["train_loss"]
        test_loss = loss["test_loss"]
        scheduler.step(test_loss)
        # current_lr = scheduler.get_last_lr()[0]
        # train_coarse_distance = np.array(train_coarse_distance)
        # train_fine_distance = np.array(train_fine_distance)
        # test_coarse_distance = np.array(test_coarse_distance)
        # test_fine_distance = np.array(test_fine_distance)

        Record_tensorboard(writer, loss, distance, epoch)
        # 每隔N个epoch保存模型的检查点（在主进程上执行）
        save_interval = 5  # 设置保存检查点的间隔，例如每隔50个epoch保存一次
        if epoch % save_interval == 0:
            checkpoint_path = f'{model_saved}checkpoint_epoch_{epoch}.pt'  # 添加model_saved路径前缀
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gat_model_state_dict': gat_model.state_dict(),  # 保存 GAT 模型的状态
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

    writer.flush()

    print("------------------------------------------------  end  --------------------------------------------")


# 主函数
def train_and_eval(model, gat_model, trainloader, testloader, optimizer, loss_function, device, scaler, config):
    # ----------------- 训练阶段 -----------------
    train_loss, train_coarse_loss, train_fine_loss, train_distances_coarse, train_distances_fine = run_epoch(
        model, gat_model, trainloader, optimizer, loss_function, device, scaler, config, is_training=True
    )

    # ----------------- 测试阶段 -----------------
    test_loss, val_coarse_loss, val_fine_loss, test_distances_coarse, test_distances_fine = run_epoch(
        model, gat_model, testloader, None, loss_function, device, scaler, config, is_training=False
    )
    loss_dict = {
        "train_loss": train_loss,
        "train_coarse_loss": train_coarse_loss,
        "train_fine_loss": train_fine_loss,
        "test_loss": test_loss,
        "val_coarse_loss": val_coarse_loss,
        "val_fine_loss": val_fine_loss,
    }
    distances_dict = {
        "train_distances_coarse": train_distances_coarse,
        "train_distances_fine": train_distances_fine,
        "test_distances_coarse": test_distances_coarse,
        "test_distances_fine": test_distances_fine
    }
    return loss_dict, distances_dict


def run_epoch(model, gat_model, dataloader, optimizer, loss_function, device, scaler, config, is_training=True):
    total_loss = 0.0
    total_coarse_loss = 0.0  # 新增：粗粒度损失累加器
    total_fine_loss = 0.0  # 新增：细粒度损失累加器

    distances_coarse_total = []
    distances_fine_total = []
    # 设置模型模式
    model.train(is_training)
    if gat_model is not None:
        gat_model.train(is_training)

    # 仅在训练阶段使用混合精度和梯度更新
    autocast_context = torch.cuda.amp.autocast(enabled=is_training)
    grad_context = torch.enable_grad() if is_training else torch.no_grad()

    with grad_context:
        for batch in tqdm(dataloader, desc='Training' if is_training else 'Testing'):
            x, y, image_info = batch
            x, y = x.to(device), y.to(device)

            if is_training:
                optimizer.zero_grad()

            with autocast_context:
                # ----------------- 公共前向计算 -----------------
                coarse_feature, y_pred = model(x)

                # 计算粗粒度损失
                coarse_loss = loss_function(y_pred, y, config.num_landmark, config.batch_size)

                # 坐标计算（统一接口）
                distances_coarse, coordinate, target, scale, coordinate_scaled = calculate_distance(
                    x, y, y_pred, image_info, config, device)

                # ----------------- 训练阶段特有逻辑 -----------------
                # GAT细粒度预测
                data_crop = get_patch(image_info["origin"], coordinate, image_info["image_size"], config.crop_size,
                                      device)
                fine_predict = gat_model(data_crop, coordinate_scaled, coarse_feature)
                distances_fine = get_error_fine(fine_predict, target, scale, config.device)
                # 细粒度损失
                criterion = nn.SmoothL1Loss(reduction='mean')
                loss_func = AdaptiveKeypointUncertaintyLoss(
                    num_keypoints=config.num_landmark,
                    reduction='mean',
                    device = config.device
                )
                fine_loss = loss_func(fine_predict, coordinate_scaled)

                # 总损失（加权）
                loss = (1 - config.alpha) * fine_loss + config.alpha * coarse_loss * config.prop

            # ----------------- 训练阶段特有反向传播 -----------------
            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # 损失累加（考虑最后一个batch可能不满）
            total_loss += loss.item() * x.size(0)
            total_coarse_loss += coarse_loss.item() * x.size(0)  # 新增：粗粒度损失累加
            total_fine_loss += fine_loss.item() * x.size(0)  # 新增：细粒度损失累加
            distances_coarse_total.append(distances_coarse)
            distances_fine_total.append(distances_fine)

    # 计算平均指标
    avg_loss = total_loss / len(dataloader.dataset)
    avg_coarse_loss = total_coarse_loss / len(dataloader.dataset)  # 新增：粗粒度平均损失
    avg_fine_loss = total_fine_loss / len(dataloader.dataset)  # 新增：细粒度平均损失
    distances_coarse_total = torch.cat(distances_coarse_total, dim=0)
    distances_fine_total = torch.cat(distances_fine_total, dim=0)
    avg_distances_coarse = torch.mean(distances_coarse_total, dim=0)
    avg_distances_fine = torch.mean(distances_fine_total, dim=0)

    # 返回新增的损失指标
    return avg_loss, avg_coarse_loss, avg_fine_loss, avg_distances_coarse, avg_distances_fine


def calculate_distance(x, y, y_pred, image_info, config, device):
    """
    计算每个点的距离并返回所有距离
    """
    distances = []
    image_size = config.image_size

    for i in range(x.size(0)):
        y_single = y[i].unsqueeze(0)
        y_pred_single = y_pred[i].unsqueeze(0)

        y_max_indices = find_max_indices(y_single, config.num_landmark)
        y_pred_max_indices = find_max_indices(y_pred_single, config.num_landmark)

        """
            1.获取每个样本的原始图像大小
            2.获取一阶段预测坐标（像素坐标）
            3.以固定size截取
        """
        origin_size = torch.stack([
            image_info["image_size"][0][i],
            image_info["image_size"][1][i],
            image_info["image_size"][2][i]
        ])

        # origin_size = torch.stack([image_info[1][0][i], image_info[1][1][i], image_info[1][2][i]])

        # 计算点的坐标误差
        distance, predicted_pixel, target_pixel, scale = get_error(y_max_indices, y_pred_max_indices, image_size,
                                                                   origin_size, device, image_info)
        # 生成二阶段数据
        # data = get_patch(image_info["origin"],predicted_pixel)
        # distances.append(distance.detach().cpu())
    return distance, predicted_pixel, target_pixel, scale, y_pred_max_indices.unsqueeze(0)


def get_patch(image, center, image_size, crop_size, device):
    """
    从给定的图像中截取一个指定大小的 3D patch。

    Args:
        image (tensor): 图像。
        center (tuple): patch 的中心坐标 (x, y, z)。sd
        image_size (tuple): patch 的大小 (size_x, size_y, size_z)。

    Returns:
        sitk.Image: 截取的 3D patch landmark数的3d patch。
    """
    # 读取图像
    patch_size = torch.tensor([crop_size, crop_size, crop_size], dtype=torch.int32).to(device)
    image_size = torch.stack(image_size).squeeze().to(device)

    # 假设 image 是一个批次图像，形状为 (batch_size, channels, depth, height, width)
    # center 是一个列表，包含每个图像的中心点坐标列表

    # 初始化一个空列表，用于存储所有图像的 patch
    batch_patch_list = []

    # 遍历批次中的每一张图像
    for i in range(image.size(0)):
        # 将当前图像转换为 SimpleITK 格式
        image_sitk = sitk.GetImageFromArray(image[i].permute(2, 1, 0).numpy())

        # 获取当前图像的中心点列表
        center_i = center[i]
        # start_coords = []

        # 初始化一个空列表，用于存储当前图像的所有 patch
        image_patch_list = []

        # 遍历当前图像的每个中心点坐标
        for center_coords in center_i:
            # 计算起始坐标
            start = (center_coords - patch_size // 2).int()

            # 检查每个维度是否越界，并调整起始坐标
            for j in range(3):
                if start[j] < 0:
                    start[j] = 0
                if start[j] + patch_size[j] > image_size[j]:
                    start[j] = image_size[j] - patch_size[j]

            # start_coords.append(start.tolist())

            # 裁剪图像块 (Region of Interest)
            patch = sitk.RegionOfInterest(
                image_sitk,
                patch_size.tolist(),
                start.tolist()
            )
            # 将裁剪后的 patch 转换为 PyTorch 张量，并调整维度顺序
            patch_tensor = torch.from_numpy(sitk.GetArrayFromImage(patch)).permute(2, 1, 0).float()
            patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
            patch_tensor = F.interpolate(patch_tensor, size=(32, 32, 32), mode='trilinear',
                                         align_corners=False).squeeze(0)
            # 将当前 patch 添加到当前图像的 patch 列表中
            image_patch_list.append(patch_tensor)

        # 将当前图像的所有 patch 堆叠成一个张量，形状为 (num_patches, channels, depth, height, width)
        image_patch_tensor = torch.stack(image_patch_list, dim=0)

        # 将当前图像的 patch 张量添加到批次列表中
        batch_patch_list.append(image_patch_tensor)

    # 将整个批次的 patch 堆叠成一个张量，形状为 (batch_size, num_patches, channels, depth, height, width)
    batch_patch_tensor = torch.stack(batch_patch_list, dim=0)

    return batch_patch_tensor


def process_data(image_path, label_path, config):
    # 读取图像
    filename = os.path.basename(image_path).split('.')[0]
    landmarkname = os.path.basename(label_path).split('.')[0]
    if filename != landmarkname:
        # 如果不一致，抛出异常
        raise ValueError(f"Filename mismatch: {filename} does not match {landmarkname}")
    image_ct = sitk.ReadImage(image_path, sitk.sitkInt16)
    spacing = image_ct.GetSpacing()
    spacing = torch.tensor(spacing)
    origin = image_ct.GetOrigin()
    origin = torch.tensor(origin)
    image_array = sitk.GetArrayFromImage(image_ct)

    # 调整图像大小
    # (z,y,x)
    # image_array = image_array.astype(np.float32)
    image_array = image_array.transpose((2, 1, 0))
    # (x,y,z)
    image_size = image_array.shape
    scale = [config.image_size[i] / image_size[i] for i in range(3)]
    image_array = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    origin_image = torch.FloatTensor(image_array).squeeze(0).squeeze(0)
    image_resized = F.interpolate(image_array, size=config.image_size, mode='trilinear', align_corners=True)
    image_resized = image_resized.squeeze(0)
    # image_resized = image_resized.astype(np.float32)

    # 读取标签并调整大小
    label = JsonLoader(label_path, spacing, origin)
    scaled_label = label * torch.tensor(scale, dtype=torch.float32)
    label_resized = generate_heatmap_target1(list(config.heatmap_size), scaled_label, config.heatmap_sigma,
                                             scale=config.heatmap_scale, normalize=True)
    # return image_resized, label_resized
    image_info = {
        "filename": filename,
        "image_size": image_size,
        "spacing": spacing,
        "origin": origin_image
    }
    return image_resized, label_resized, image_info


def find_max_indices(tensor, num_landmark):
    # 找预测的热力图的最大值
    max_index_list = []
    tensor = torch.squeeze(tensor)
    for i in range(num_landmark):
        tensor_landmark = tensor[i, :, :, :]
        max_indices_flat = torch.argmax(tensor_landmark)
        shape = tensor_landmark.shape
        max_index = unravel_index(max_indices_flat, shape)
        max_index_list.append(max_index)
    max_index_tensor = torch.stack(max_index_list, dim=0)
    return max_index_tensor


def JsonLoader(label_path, space, origin):
    labelList = [
        "OR", "OL", "S", "N", "ANS", "PNS",
        "Ba", "PoR", "PoL", "A", "Pog", "Me",
        "Gn", "B", "GoR", "GoL", "UI", "UIa",
        "Spr", "LI", "LIa", "Id", "CoR", "CoL"]
    label_to_position = {}

    with open(label_path, 'r') as file:
        data = json.load(file)

    for markup in data['markups']:
        for control_point in markup['controlPoints']:
            label_to_position[control_point['label']] = control_point['position']
    positionList = [label_to_position[label] for label in labelList if label in label_to_position]

    # 将点的列表转换为PyTorch张量
    tensor_points = torch.tensor(positionList, dtype=torch.float32)

    """读取.csv格式landmark,并将其RAS坐标系转换成世界坐标系对应体素"""
    # get image spacing

    # get image origin
    # change origin
    indices = [0, 1]
    origin[indices] = origin[indices]

    # transform matrix
    matrix = torch.FloatTensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])

    # load landmark
    # 根据公式转换
    coordinate = transform_coordinate(tensor_points, origin, space, matrix)

    return coordinate


def transform_coordinate(ras, origin, spacing, matrix):
    """转换公式"""
    ras = ras - origin
    # 转换矩阵 逆
    matrix_inverse = torch.inverse(matrix)
    # 执行矩阵乘法
    ras = torch.matmul(matrix_inverse, ras.T).T
    coordinate = torch.divide(ras, spacing)
    coordinate_round = torch.tensor(coordinate)

    coordinate_round = torch.abs(coordinate_round)
    return coordinate_round


def get_error(target, predict, image_size, origin_size, device, image_info):
    scale = [origin_size[i] / image_size[i] for i in range(3)]
    scale = torch.tensor(scale).to(device)
    scale = scale.unsqueeze(0).unsqueeze(0)

    origin = image_info["origin"][0].to(device)

    spacing = torch.tensor([0.39, 0.39, 0.39]).to(device)
    scaled_predict = predict * scale * spacing
    scale_label = target * scale * spacing

    matrix = torch.FloatTensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]]).to(device)
    filename = image_info["filename"][0]

    predict_pixel = predict * scale
    target_pixel = target * scale
    # predict_coordinate = transform_coordinate_reverse(predict_pixel,origin,spacing,matrix,filename)
    # 只有batchsize为1有用
    distance = torch.sqrt(torch.sum(torch.pow(scaled_predict.squeeze() - scale_label.squeeze(), 2), 1)).unsqueeze(0)
    return distance, predict_pixel, target_pixel, scale


def transform_coordinate_reverse(coordinate, origin, spacing, matrix, filename):
    """将ijk坐标系转换成ras坐标系"""

    matrix_inverse = torch.inverse(matrix)
    ras = coordinate * spacing
    ras = torch.matmul(matrix_inverse, ras.T).T
    ras = ras + origin
    ras = torch.as_tensor(ras, dtype=torch.float32)

    return ras


def Record_tensorboard(writer, loss, distance, epoch):
    train_loss = loss["train_loss"]
    test_loss = loss["test_loss"]

    train_coarse_loss = loss["train_coarse_loss"]
    train_fine_loss = loss["train_fine_loss"]

    val_coarse_loss = loss["val_coarse_loss"]
    val_fine_loss = loss["val_fine_loss"]

    # 计算各项指标的平均值（假设输入为张量）
    train_coarse_mean = torch.mean(distance["train_distances_coarse"]).item()  # 粗粒度训练误差均值
    train_fine_mean = torch.mean(distance["train_distances_fine"]).item()  # 细粒度训练误差均值
    test_coarse_mean = torch.mean(distance["test_distances_coarse"]).item()  # 粗粒度测试误差均值
    test_fine_mean = torch.mean(distance["test_distances_fine"]).item()  # 细粒度测试误差均值

    # 记录损失到TensorBoard
    writer.add_scalar('Loss/train/overall', train_loss, epoch)  # 训练集总体损失
    writer.add_scalar('Loss/test/overall', test_loss, epoch)  # 测试集总体损失
    writer.add_scalar('Loss/train/coarse', train_coarse_loss, epoch)  # 训练集粗粒度损失
    writer.add_scalar('Loss/train/fine', train_fine_loss, epoch)  # 训练集细粒度损失
    writer.add_scalar('Loss/val/coarse', val_coarse_loss, epoch)  # 验证集粗粒度损失
    writer.add_scalar('Loss/val/fine', val_fine_loss, epoch)  # 验证集细粒度损失

    # 记录平均误差到TensorBoard
    writer.add_scalar('Error/train/coarse_mean', train_coarse_mean, epoch)  # 训练集粗粒度误差均值
    writer.add_scalar('Error/train/fine_mean', train_fine_mean, epoch)  # 训练集细粒度误差均值
    writer.add_scalar('Error/test/coarse_mean', test_coarse_mean, epoch)  # 测试集粗粒度误差均值
    writer.add_scalar('Error/test/fine_mean', test_fine_mean, epoch)  # 测试集细粒度误差均值
    print("\n")
    # 打印日志（包含所有指标）
    print(f'epoch: {epoch} | '
          f'train_loss: {train_loss:.4f} | '
          f'test_loss: {test_loss:.4f} | '
          f'train_coarse_error: {train_coarse_mean:.2f} | '
          f'train_fine_error: {train_fine_mean:.2f} | '
          f'test_coarse_error: {test_coarse_mean:.2f} | '
          f'test_fine_error: {test_fine_mean:.2f}')


def get_error_fine(predict, target, scale, device):
    spacing = torch.tensor([0.39, 0.39, 0.39]).to(device)
    predict = predict * scale * spacing
    target = target * spacing

    # predict_coordinate = transform_coordinate_reverse(predict_pixel,origin,spacing,matrix,filename)
    # 只有batchsize为1有用
    distance = torch.sqrt(torch.sum(torch.pow(predict.squeeze() - target.squeeze(), 2), 1)).unsqueeze(0)
    return distance


def fix_random_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    fix_random_seeds(1234)
    main()
