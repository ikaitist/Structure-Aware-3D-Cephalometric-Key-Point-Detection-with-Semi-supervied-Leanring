import torch.nn as nn
import torch
import os
import json
import numpy as np
from torchvision.transforms import Normalize


def loss_function(predict, target, nums_landmarks, batchsize):
    loss_mse = nn.MSELoss()
    # predict_norm = torch.zeros_like(predict)
    # for j in range(batchsize):
    #     landmark_b = predict[j]
    #     # 对每个 landmark 的 (x, y, z) 进行最小-最大归一化
    #     for k in range(nums_landmarks):
    #         landmark_n = landmark_b[k]
    #         min_val = landmark_n.min()
    #         max_val = landmark_n.max()
    #         # 应用最小-最大归一化
    #         predict_norm[j, k] = (landmark_n - min_val) / (max_val - min_val)
    loss = loss_mse(target, predict)
    return loss


def find_max_indices1(tensor):
    max_indices_list = []

    for i in range(5):  # 假设有5个landmarks
        # 找到值为1的区域
        ones_indices = torch.nonzero(tensor[:, i, :, :, :] == 1, as_tuple=False)

        if len(ones_indices) > 0:
            # 计算几何中心
            centroid = torch.mean(ones_indices.float(), dim=0)
            max_indices_list.append(centroid.int())
        else:
            # 如果没有找到值为1的区域，则添加占位符（例如：-1）
            max_indices_list.append(torch.tensor([-1, -1, -1, -1], dtype=torch.int32))

    max_indices_tensor = torch.stack(max_indices_list)
    max_indices_tensor = max_indices_tensor[:, 1:]
    return max_indices_tensor


def find_max_indices(tensor):
    max_indices_list = []
    tensor = torch.squeeze(tensor)
    for i in range(5):
        max_value = torch.max(tensor[i, :, :, :])
        max_indices = torch.nonzero(tensor[i, :, :, :] == max_value, as_tuple=False)
        max_indices_list.append(max_indices[0])
    max_indices_tensor = torch.stack(max_indices_list, dim=0)
    return max_indices_tensor


def Record_tensorboard(writer, train_loss, test_loss, train_distance, test_distance, epoch):
    # writer.add_scalar('learning rate', current_lr, epoch)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    # writer.add_scalar('chin_R', train_distance[0], epoch)
    # writer.add_scalar('mand_r_R', train_distance[1], epoch)
    # writer.add_scalar('mand_l_R', train_distance[2], epoch)
    # writer.add_scalar('odont_proc_R', train_distance[3], epoch)
    # writer.add_scalar('occ_bone_R', train_distance[4], epoch)
    # writer.add_scalar('chin_E', test_distance[0], epoch)
    # writer.add_scalar('mand_r_E', test_distance[1], epoch)
    # writer.add_scalar('mand_l_E', test_distance[2], epoch)
    # writer.add_scalar('odont_proc_E', test_distance[3], epoch)
    # writer.add_scalar('occ_bone_E', test_distance[4], epoch)
    average_train = (train_distance[0] + train_distance[1] + train_distance[2] + train_distance[3] + train_distance[
        4]) / 5
    writer.add_scalar('average_R', average_train, epoch)
    average_test = (test_distance[0] + test_distance[1] + test_distance[2] + test_distance[3] + test_distance[
        4]) / 5
    writer.add_scalar('average_E', average_test, epoch)
    print('epoch: ', epoch,
          'train_loss: ', train_loss,
          'test_loss: ', test_loss,
          'average_error_train', average_train,
          'average_error_test', average_test
          )


def StoreToJson(landmarks, filename):
    filename = filename + ".json"
    # 对应的点名称
    file_dir = "/home/user/imp/data/data_unlabeled/landmark"
    landmarks_names = ["chin", "mand_r", "mand_l", "odont_proc", "occ_bone"]

    # 转换为Python的列表格式，并保留所需的元数据
    data = {
        "markups": [
            {
                "controlPoints": []
            }
        ]
    }
    # 遍历点并添加到控制点列表中
    for i, (point, name) in enumerate(zip(landmarks, landmarks_names)):
        point_dict = {
            "id": str(i + 1),
            "label": name,
            "position": point.cpu().tolist(),
        }
        data["markups"][0]["controlPoints"].append(point_dict)

    # 确保目录存在
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # 构造完整文件路径
    file_path = os.path.join(file_dir, filename)

    # 将数据写入JSON文件
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # print(f"Points have been saved to {file_path}")


def get_error(target, predict, image_size, origin_size, device):
    scale = [origin_size[i] / image_size[i] for i in range(3)]
    scale = torch.tensor(scale).to(device)
    scale = scale.unsqueeze(0).unsqueeze(0)
    scaled_predict = predict * scale
    scale_label = target * scale
    distance = torch.sqrt(torch.sum(torch.pow(scaled_predict.squeeze() - scale_label.squeeze(), 2), 1)).unsqueeze(0)
    return distance


def save_coordinate_real(coordinate, number):
    # 初始的 JSON 数据
    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 5,
                "controlPoints": [
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_0",
                        "label": "chin",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [7.16592, -116.932, 223.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_1",
                        "label": "mand_r",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [-41.6039, -41.1211, 289.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_2",
                        "label": "mand_l",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [51.5899, -41.6039, 292.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_3",
                        "label": "odont_proc",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [5.39975, -32.4642, 259.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_4",
                        "label": "occ_bone",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [6.72, 52.748, 269.11],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    }
                ],
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

    # 更新 controlPoints 的 position
    for i, point in enumerate(data['markups'][0]['controlPoints']):
        point['position'] = coordinate[i].cpu().numpy().tolist()
    output_path = f'/data1/lyh/landmark_json/{number}_real.json'
    # 将更新后的数据写入 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def save_coordinate_predict(coordinate, number):
    # 初始的 JSON 数据
    data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 5,
                "controlPoints": [
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_0",
                        "label": "chin",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [7.16592, -116.932, 223.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_1",
                        "label": "mand_r",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [-41.6039, -41.1211, 289.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_2",
                        "label": "mand_l",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [51.5899, -41.6039, 292.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_3",
                        "label": "odont_proc",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [5.39975, -32.4642, 259.0],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    },
                    {
                        "id": "vtkMRMLMarkupsFiducialNode_4",
                        "label": "occ_bone",
                        "description": "",
                        "associatedNodeID": "vtkMRMLScalarVolumeNode2",
                        "position": [6.72, 52.748, 269.11],
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    }
                ],
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

    # 更新 controlPoints 的 position
    for i, point in enumerate(data['markups'][0]['controlPoints']):
        point['position'] = coordinate[i].cpu().numpy().tolist()
    output_path = f'/data1/lyh/landmark_json/{number}_predict.json'
    # 将更新后的数据写入 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)



def get_global_feature(ROIs, coarse_feature, landmarkNum):
    X1, Y1, Z1 = ROIs[:, :, 0], ROIs[:, :, 1], ROIs[:, :, 2]

    L, H, W = coarse_feature.size()[-3:]
    X1 = torch.round(X1 * (H - 1)).type(torch.int32)
    Y1 = torch.round(Y1 * (W - 1)).type(torch.int32)
    Z1 = torch.round(Z1 * (L - 1)).type(torch.int32)
    # print("Z1 values:", Z1)
    # print("Y1 values:", Y1)
    # print("X1 values:", X1)
    global_embedding = torch.cat([coarse_feature[:, :, Z1[0, i], Y1[0, i], X1[0, i]] for i in range(landmarkNum)],
                                 dim=0).unsqueeze(0)
    return global_embedding