import csv

import torch
import SimpleITK as sitk
import numpy as np
import torch.nn as nn
import json
from dataset import get_file_list_gz, get_file_list_unlabeled
from dataset import lable_loader
from torch.utils.data import Dataset, DataLoader
from model import Unet3d
# from tests.model_scn import Unet3d
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

# param = 'Baseline_Unet_E_300_L=0.0001_S=128_factor=0.95_3.15_con'
# param = 'SwinUnetr-gat_E_200_L=0.0001_S=128_factor=0.95_cropSize=16_alpha=0.4_prop=10000_test1'
# param = "Baseline_SwinUnetr_E_300_L=0.0015_S=128_factor=0.95_2.25"
param = "SwinUnetr-gat_E_200_L=0.0015_S=128_factor=0.95_cropSize=16_alpha=0.4_prop=10000_BestR"

# param = "SwinUnetr-gat_E_200_L=0.0001_S=128_factor=0.95_cropSize=24_alpha=0.4_prop=10000_con"
# param = "SwinUnetr-gat_E_200_L=0.0015_S=128_factor=0.95_cropSize=16_alpha=0.9_prop=10000_3.11"

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
available_devices = [2]  # 指定可用的GPU设备编号
data = './data'
save_path = cache + '/result/'

class Config(object):
    # File paths
    data_train_dir = data + "/data_k/Train"
    data_test_dir = data + "/data_k/Test"
    trainNet = "SwinUNETR"
    # trainNet = "Unet"
    # lable_dir = 'data'
    image_list = []
    label_list = []
    epoch = 300
    batch_size = 1
    # image_size = [128, 128, 128]
    image_size = [128, 128, 128]
    heatmap_size = image_size
    heatmap_scale = 1000
    num_landmark = 24
    heatmap_sigma = torch.nn.Parameter(torch.full((num_landmark,), 4.0))
    leaning_rate = 0.0015


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
    if (config.trainNet == "SwinUNETR"):
        model = SwinUNETR(
            in_channels=1,
            out_channels=config.num_landmark,
            img_size=(128, 128, 128),
            feature_size=48, ).to(device)
    else:
        model = Unet3d(in_channels=1, n_classes=config.num_landmark, n_channels=32).to(device)  # 网络


    pretrained_model_path = cache + '/model_saved/' + param + '/checkpoint_epoch_44.pt'
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


    print(
        "------------------------------------------------start training--------------------------------------------")
    output = train(model, datasets_test, device, config)
    print(output)

print("------------------------------------------------  end  --------------------------------------------")

def calculate_metrics(errors):
    metrics = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'rate_2mm': np.mean(errors <= 2) * 100,
        'rate_2.5mm': np.mean(errors <= 2.5) * 100,
        'rate_3mm': np.mean(errors <= 3) * 100,
        'rate_4mm': np.mean(errors <= 4) * 100,
        'rate_8mm': np.mean(errors <= 8) * 100
    }
    return {k: round(float(v), 2) for k, v in metrics.items()}


def train(model, testloader, device, config):
    #  --------------------------------------------------------------------------------------------------------------
    model.eval()
    all_errors = []  # 存储所有点的误差
    per_landmark_errors = {}  # 按点存储误差列表
    for i in range(config.num_landmark):
        per_landmark_errors[i] = []

    with torch.no_grad():
        for batch in testloader:
            x, y, image_info = batch
            x, y = x.to(device), y.to(device)
            with autocast():
                _, y_pred = model(x)

            # 计算误差
            y_max_indices = find_max_indices(y, config.num_landmark)
            y_pred_max_indices = find_max_indices(y_pred, config.num_landmark)
            origin_size = torch.stack([
                image_info["image_size"][0][0],
                image_info["image_size"][1][0],
                image_info["image_size"][2][0]
            ])

            distance, predicted = get_error(y_max_indices, y_pred_max_indices,
                                            config.image_size, origin_size,
                                            device, image_info)

            # 将结果移到CPU并转为numpy（保持设备一致性）
            batch_errors = distance.cpu().numpy().flatten()
            all_errors.extend(batch_errors)

            # 按点存储误差
            for landmark_idx in range(config.num_landmark):
                per_landmark_errors[landmark_idx].append(batch_errors[landmark_idx])

            store_json(predicted, image_info["filename"][0])

    # 转换为numpy数组保持设备一致性
    all_errors = np.array(all_errors)

    # 计算总统计
    total_metrics = calculate_metrics(all_errors)

    # 计算各点统计
    landmark_metrics = {}
    for landmark_idx in range(config.num_landmark):
        landmark_metrics[landmark_idx] = calculate_metrics(
            np.array(per_landmark_errors[landmark_idx])
        )

    # 打印结果
    print("\n===== Overall Metrics =====")
    print(f"Mean Error: {total_metrics['mean']}±{total_metrics['std']}mm")
    print(f"2mm检出率: {total_metrics['rate_2mm']}%")
    print(f"2.5mm检出率: {total_metrics['rate_2.5mm']}%")
    print(f"3mm检出率: {total_metrics['rate_3mm']}%")
    print(f"4mm检出率: {total_metrics['rate_4mm']}%")
    print(f"8mm检出率: {total_metrics['rate_8mm']}%")

    print("\n===== Per Landmark Metrics =====")
    for landmark_idx in range(config.num_landmark):
        metrics = landmark_metrics[landmark_idx]
        # print(f"Landmark {landmark_idx + 1}:")
        # print(f"  Mean: {metrics['mean']}±{metrics['std']}mm")
        # print(f"  ≤2mm: {metrics['rate_2mm']}%")
        # print(f"  ≤2.5mm: {metrics['rate_2.5mm']}%")
        # print(f"  ≤3mm: {metrics['rate_3mm']}%")
        # print(f"  ≤4mm: {metrics['rate_4mm']}%")
        # print(f"  ≤8mm: {metrics['rate_8mm']}%")
        # print("-----------------------------")
        labelList = [
            "OR", "OL", "S", "N", "ANS", "PNS",
            "Ba", "PoR", "PoL", "A", "Pog", "Me",
            "Gn", "B", "GoR", "GoL", "UI", "UIa",
            "Spr", "LI", "LIa", "Id", "CoR", "CoL"]
        save_to_csv(total_metrics, landmark_metrics, labelList)
    return 0


def save_to_csv(total_metrics, landmark_metrics, labelList):
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 生成完整文件路径
    csv_path = os.path.join(save_path, "landmark_errors.csv")

    # CSV表头
    header = [
        "Landmark Name", "Mean Error (mm)", "Std (mm)",
        "≤2mm (%)", "≤2.5mm (%)", "≤3mm (%)",
        "≤4mm (%)", "≤8mm (%)"
    ]

    # 写入数据
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(header)

        # 写入各关键点数据
        for idx in range(len(labelList)):
            metrics = landmark_metrics[idx]
            row = [
                labelList[idx],
                f"{metrics['mean']:.2f}",
                f"{metrics['std']:.2f}",
                f"{metrics['rate_2mm']:.2f}",
                f"{metrics['rate_2.5mm']:.2f}",
                f"{metrics['rate_3mm']:.2f}",
                f"{metrics['rate_4mm']:.2f}",
                f"{metrics['rate_8mm']:.2f}"
            ]
            writer.writerow(row)

        # 添加总体统计
        writer.writerow([])  # 空行分隔
        writer.writerow(["Overall Statistics"])
        writer.writerow([
            "All Landmarks",
            f"{total_metrics['mean']:.2f}",
            f"{total_metrics['std']:.2f}",
            f"{total_metrics['rate_2mm']:.2f}",
            f"{total_metrics['rate_2.5mm']:.2f}",
            f"{total_metrics['rate_3mm']:.2f}",
            f"{total_metrics['rate_4mm']:.2f}",
            f"{total_metrics['rate_8mm']:.2f}"
        ])

def store_json(predicted, image_info):
    file_path = '/data/8T/lyh/data/data_show/'+image_info +'_pred.json'
    predicted = predicted.cpu().numpy()
    labelList = [
        "OR", "OL", "S", "N", "ANS", "PNS",
        "Ba", "PoR", "PoL", "A", "Pog", "Me",
        "Gn", "B", "GoR", "GoL", "UI", "UIa",
        "Spr", "LI", "LIa", "Id", "CoR", "CoL"]
    json_template = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(predicted),
                "controlPoints": [],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.4, 1.0, 1.0],
                    "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                    "activeColor": [0.4, 1.0, 0.0],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": True,
                    "textScale": 3.0,
                    "glyphType": "Sphere3D",
                    "glyphScale": 3.0,
                    "glyphSize": 5.0,
                    "useGlyphScale": True,
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "translationHandleVisibility": True,
                    "rotationHandleVisibility": True,
                    "scaleHandleVisibility": True,
                    "interactionHandleScale": 3.0,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }
    for i,pred in enumerate(predicted):
        control_point = {
            "id" : str(i),
            "label": labelList[i],
            "description": "",
            "associatedNodeID": "",
            "position": pred.tolist(),
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],  # assuming fixed orientation
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined"
        }
        # print(control_point)
        json_template["markups"][0]["controlPoints"].append(control_point)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_template, json_file, ensure_ascii=False, indent=4)
    # print(json_template)


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
    scaled_label = label * torch.tensor(scale, dtype=torch.float32)
    label_resized = generate_heatmap_target1(list(config.heatmap_size), scaled_label, config.heatmap_sigma,
                                             scale=config.heatmap_scale, normalize=True)
    # return image_resized, label_resized
    image_info = {
        "filename": filename,
        "image_size": image_size,
        "spacing": spacing,
        "origin": origin
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
    return coordinate


def transform_coordinate(ras, origin, spacing, matrix):
    """将ras坐标系转换成ijk坐标系"""

    ras = ras - origin
    # 转换矩阵 逆
    matrix_inverse = torch.inverse(matrix)
    # 执行矩阵乘法
    ras = torch.matmul(matrix_inverse, ras.T).T
    coordinate = torch.divide(ras, spacing)
    coordinate_round = torch.tensor(coordinate)
    coordinate_round = torch.abs(coordinate_round)
    return coordinate_round

def transform_coordinate_reverse(coordinate, origin, spacing, matrix, filename):
    """将ijk坐标系转换成ras坐标系"""
    # print(filename)
    matrix_inverse = torch.inverse(matrix)
    ras = coordinate * spacing
    ras = torch.matmul(matrix_inverse, ras.T).T
    ras = ras + origin
    ras = torch.as_tensor(ras, dtype=torch.float32)

    return ras

def get_error(target, predict, image_size, origin_size, device , image_info):
    # print(origin_size[0])
    scale = [origin_size[i] / image_size[i] for i in range(3)]
    scale = torch.tensor(scale).to(device)

    origin = image_info["origin"][0].to(device)

    print(image_info)

    spacing = torch.tensor([0.39, 0.39, 0.39]).to(device)
    scaled_predict = predict * scale * spacing
    scale_label = target * scale * spacing

    matrix = torch.FloatTensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]]).to(device)
    filename = image_info["filename"][0]
    predicted = predict * scale
    predict_coordinate = transform_coordinate_reverse(predicted,origin,spacing,matrix,filename)

    distance = torch.sqrt(torch.sum(torch.pow(scaled_predict.squeeze() - scale_label.squeeze(), 2), 1)).unsqueeze(0)
    return distance,predict_coordinate


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
