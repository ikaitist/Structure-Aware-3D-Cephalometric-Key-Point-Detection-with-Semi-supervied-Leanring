import torch
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
import json
from dataset import get_file_list_gz, get_file_list_unlabeled
from dataset import lable_loader
from torch.utils.data import Dataset, DataLoader
# from model import Unet3d
from tests.model_scn import Unet3d
from tqdm import tqdm
from torch.optim import lr_scheduler
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

param = 'Baseline_SCN_E_300_L=0.0001_S=128_factor=0.95_3.16_con'
print(param)
# baselin=>'SwinUnetr_lr=1e-4_size96_seg_ReduceLROnPlateau_2_batchsize_1'
# 创建用于存储日志的文件夹
cache = './cache'
logs_folder = cache + '/logs/' + param
os.makedirs(logs_folder, exist_ok=True)
writer = SummaryWriter(logs_folder)
# 创建用于存储模型的文件夹
model_saved = cache + '/model_saved/' + param + '/'
os.makedirs(model_saved, exist_ok=True)
available_devices = [1]  # 指定可用的GPU设备编号
data = './data'


class Config(object):
    # File paths
    data_train_dir = data + "/data_k/Train"
    data_test_dir = data + "/data_k/Test"
    # trainNet = "SwinUNETR"
    trainNet = "Unet"
    # lable_dir = 'data'
    image_list = []
    label_list = []
    epoch = 301
    batch_size = 1
    image_size = [128, 128, 128]
    # image_size = [96,96,96]
    heatmap_size = image_size
    heatmap_scale = 1000
    num_landmark = 24
    heatmap_sigma = torch.nn.Parameter(torch.full((num_landmark,), 4.0))
    leaning_rate = 0.0001

class Datasets_train(Dataset):
    def __init__(self, image_list, label_list, transformer):
        self.label_list = label_list
        self.image_list = image_list
        self.config = Config()

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        image, label, filename = process_data(image_path, label_path, self.config)
        return image, label, filename

    def __len__(self):
        return len(self.image_list)


class Datasets_test(Dataset):
    def __init__(self, image_list, label_list, transformer):
        self.label_list = label_list
        self.image_list = image_list
        self.transformer = transformer
        self.config = Config()

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        image, label, filename = process_data(image_path, label_path, self.config)
        return image, label, filename

    def __len__(self):
        return len(self.image_list)


def main():
    # selected_device = available_devices[rank]  # 根据rank选择对应的GPU设备
    # torch.cuda.set_device(selected_device)
    config = Config()
    device = torch.device('cuda', available_devices[0])
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 放入DDP

    image_train_list, label_train_list = get_file_list_gz(config.data_train_dir)  # done get file list
    image_test_list, label_test_list = get_file_list_gz(config.data_test_dir)
    transformer = 0  # 待加入
    # 加载训练集，测试集
    datasets_train = Datasets_train(image_train_list, label_train_list, transformer)
    datasets_test = Datasets_test(image_test_list, label_test_list, transformer)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(datasets_train)  # 创建一个分布式训练的数据采样器
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

    # backbone Unet3d   SwinUNETR 969696
    if (config.trainNet == "SwinUNETR"):
        model = SwinUNETR(
            in_channels=1,
            out_channels=config.num_landmark,
            img_size=(128,128,128),
            feature_size=48, ).to(device)
    else:
        model = Unet3d(in_channels=1, n_classes=config.num_landmark, n_channels=32).to(device)  # 网络
    # 放入DDP
    # # 加载预训练模型
    # model.load_state_dict(torch.load(pretrained_model_path, map_location=lambda storage, loc: storage.cuda(2)))
    # pretrained_model_path = cache + '/model_saved/Baseline_SwinUnetr_E_300_L=0.0015_S=128_factor=0.95_2.25/checkpoint_epoch_280.pt'
    # checkpoint = torch.load(pretrained_model_path,map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    loss_mse = loss_function  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.leaning_rate)  # 优化器
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
        # train test val
        train_loss, test_loss, train_distance, test_distance = train(model, datasets_train, datasets_test,
                                                                     optimizer, loss_mse,
                                                                     config.batch_size, device, scaler,
                                                                     config)  # device: gpu
        scheduler.step(test_loss)
        # current_lr = scheduler.get_last_lr()[0]
        train_distance = np.array(train_distance)
        test_distance = np.array(test_distance)
        Record_tensorboard(writer, train_loss, test_loss, train_distance, test_distance, epoch)
        # 每隔N个epoch保存模型的检查点（在主进程上执行）
        save_interval = 20  # 设置保存检查点的间隔，例如每隔50个epoch保存一次
        if epoch % save_interval == 0:
            checkpoint_path = f'{model_saved}checkpoint_epoch_{epoch}.pt'  # 添加model_saved路径前缀
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # 注意：使用model.module.state_dict()获取真正的模型参数
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')

    writer.flush()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间
    model_pram_filename = f'{model_saved}model_pram_{current_time}.pt'
    model_filename = f'{model_saved}model_{current_time}.pt'
    torch.save(model.state_dict(), model_pram_filename)
    torch.save(model, model_filename)
    print("------------------------------------------------  end  --------------------------------------------")


def train(model, trainloader, testloader, optimizer, loss_function, batch_size, device, scaler, config):
    train_loss = 0
    model.train()
    # 初始化存储每个点的距离的列表
    distances_train = []
    distance_all = np.array([], dtype=float)
    for batch in tqdm(trainloader, desc='Training'):
        # x是ct数据 y是GT热力图 image_info
        x, y, image_info = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast():
            # coarse_feature size (batchsize,channel,128 128 128)  (512,)
            _, y_pred = model(x)
            loss = loss_function(y_pred, y, config.num_landmark, batch_size)

        with torch.no_grad():
            image_size = config.image_size
            train_loss += loss.item()
            # 对于每个点计算距
            for i in range(x.size(0)):
                y_single = y[i].unsqueeze(0)
                y_pred_single = y_pred[i].unsqueeze(0)
                y_max_indices = find_max_indices(y_single, config.num_landmark)
                y_pred_max_indices = find_max_indices(y_pred_single, config.num_landmark)
                origin_size = torch.stack([image_info[1][0][i], image_info[1][1][i], image_info[1][2][i]])
                # 提取每个点的坐标
                distance = get_error(y_max_indices, y_pred_max_indices, image_size, origin_size, device)
                distances_train.append(distance.detach().cpu())

        scaler.scale(loss).backward()  # 缩放梯度
        scaler.step(optimizer)  # 更新参数
        scaler.update()  # 更新缩放因子
    train_Off = torch.cat(distances_train, dim=0)
    distance_train = torch.mean(train_Off, dim=0)
    train_loss = train_loss / len(trainloader.dataset)

    #  --------------------------------------------------------------------------------------------------------------
    test_loss = 0
    model.eval()
    distances_test = []
    with torch.no_grad():
        for batch in testloader:
            x, y, image_info = batch
            x, y = x.to(device), y.to(device)
            with autocast():
                _ , y_pred = model(x)
                loss = loss_function(y_pred, y, config.num_landmark, batch_size)
            test_loss += loss.item()
            image_size = config.image_size

            # 对于每个点计算距离
            # 对于每个点计算距
            for i in range(x.size(0)):
                y_single = y[i].unsqueeze(0)
                y_pred_single = y_pred[i].unsqueeze(0)
                y_max_indices = find_max_indices(y_single, config.num_landmark)
                y_pred_max_indices = find_max_indices(y_pred_single, config.num_landmark)

                origin_size = torch.stack([image_info[1][0][i], image_info[1][1][i], image_info[1][2][i]])
                # 提取每个点的坐标
                distance = get_error(y_max_indices, y_pred_max_indices, image_size, origin_size, device)
                distances_test.append(distance.detach().cpu())
    # distances_test 现在包含了每个点对之间的距离
    test_Off = torch.cat(distances_test, dim=0)
    distance_test = torch.mean(test_Off, dim=0)
    return train_loss, test_loss, distance_train, distance_test


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
    # image_array = image_array.astype(np.float32)
    image_array = image_array.transpose((2, 1, 0))
    image_size = image_array.shape
    scale = [config.image_size[i] / image_size[i] for i in range(3)]
    image_array = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    image_resized = F.interpolate(image_array, size=config.image_size, mode='trilinear', align_corners=True)
    image_resized = image_resized.squeeze(0)
    # image_resized = image_resized.astype(np.float32)

    # 读取标签并调整大小
    label = JsonLoader(label_path, spacing, origin)
    # label 像素坐标值GT
    # GT* 128/512=> scale_label
    # label_resized就是 label生成的高斯热力图，作为loss计算的GT
    scaled_label = label * torch.tensor(scale, dtype=torch.float32)
    label_resized = generate_heatmap_target1(list(config.heatmap_size), scaled_label, config.heatmap_sigma,
                                             scale=config.heatmap_scale, normalize=True)
    # return image_resized, label_resized
    image_info = [filename, image_size]
    return image_resized, label_resized, image_info


def find_max_indices(tensor, num_landmark):
    # 找预测的热力图的最大值
    max_index_list = []
    tensor = torch.squeeze(tensor)
    for i in range(num_landmark):
        tensor_landmark = tensor[i, :, :, :]
        max_indices_flat =torch.argmax(tensor_landmark)
        shape =  tensor_landmark.shape
        max_index = unravel_index(max_indices_flat,shape)
        max_index_list.append(max_index)
    max_index_tensor = torch.stack(max_index_list, dim=0)
    return max_index_tensor


def JsonLoader(label_path, space, origin):
    labelList = [
        "OR", "OL", "S", "N", "ANS", "PNS",
        "Ba", "PoR", "PoL", "A", "Pog", "Me",
        "Gn", "B", "GoR", "GoL", "UI", "UIa",
        "Spr", "LI", "LIa","Id", "CoR", "CoL"]
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

    # print("space", space, space.dtype)
    # get image origin
    # change origin
    indices = [0, 1]
    origin[indices] = origin[indices]
    # print("origin=", origin, origin.dtype)

    # transform matrix
    matrix = torch.FloatTensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    # print("matrix=", matrix, matrix.dtype)

    # load landmark
    # 根据公式转换
    coordinate = transform_coordinate(tensor_points, origin, space, matrix)
    # print("coordinate=", coordinate, coordinate.dtype, coordinate.shape)
    # 真实坐标-》像素坐标
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


def get_error(target, predict, image_size, origin_size, device):
    scale = [origin_size[i] / image_size[i] for i in range(3)]
    scale = torch.tensor(scale).to(device)
    scale = scale.unsqueeze(0).unsqueeze(0)
    spacing = torch.tensor([0.39, 0.39, 0.39]).to(device)
    scaled_predict = predict * scale * spacing
    scale_label = target * scale * spacing
    distance = torch.sqrt(torch.sum(torch.pow(scaled_predict.squeeze() - scale_label.squeeze(), 2), 1)).unsqueeze(0)
    return distance


def Record_tensorboard(writer, train_loss, test_loss, train_distance, test_distance, epoch):
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    average_train = sum(train_distance) / len(train_distance)
    writer.add_scalar('average_R', average_train, epoch)
    average_test = sum(test_distance) / len(test_distance)
    writer.add_scalar('average_E', average_test, epoch)
    print('epoch: ', epoch,
          'train_loss: ', train_loss,
          'test_loss: ', test_loss,
          'average_error_train', average_train,
          'average_error_test', average_test
          )


if __name__ == '__main__':
    main()
