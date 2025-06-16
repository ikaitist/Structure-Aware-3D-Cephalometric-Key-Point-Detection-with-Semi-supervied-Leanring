# 测试模型结果生成.pth文件测试

import os

import torch
import SimpleITK as sitk
import numpy as np
import torch.nn as nn

from dataset import get_file_list
from dataset import get_file_list_gz
from dataset import lable_loader
from torch.utils.data import Dataset, DataLoader
from model import Unet3d
# from tests.model_scn import Unet3d
from unter import UNETR
from utils.heatmap_image_generate import generate_heatmap_target1
from result_test import distance_count

device = torch.device("cuda:0")
# 需要修改的参数
param = 'OH_lr_0.0001_e_500_176_Unet'
save_dir = './model_test/' + param + '/'
os.makedirs(save_dir, exist_ok=True)
model_saved = './cache/model_saved/' + param + '/'
model_load = model_saved + 'checkpoint_epoch_300.pt'
data = './data'


class Config(object):
    # 需要修改的参数
    # data_train_dir = os.path.join(data, 'unet_data_resize/Train3')
    # data_test_dir = os.path.join(data, 'unet_data_resize/Test3')
    data_train_dir = os.path.join(data, 'data_gz/Train')
    data_test_dir = os.path.join(data, 'data_gz/Test')
    image_list = []
    label_list = []
    # epoch = 300
    batch_size = 1
    # 需要修改的参数
    # image_size = [256, 256, 128]
    image_size = [128, 128, 96]
    heatmap_size = image_size
    heatmap_scale = 1000
    num_landmark = 5
    heatmap_sigma = torch.nn.Parameter(torch.full((num_landmark,), 4.0))
    # leaning_rate = 0.000001


class Datasets_test(Dataset):
    def __init__(self, image_list, label_list, transformer):
        self.label_list = label_list
        self.image_list = image_list

    def __getitem__(self, index):
        config = Config()
        image = self.image_list[index]
        label = self.label_list[index]
        # get image tensor
        image_ct = sitk.ReadImage(image, sitk.sitkInt16)
        image_array = sitk.GetArrayFromImage(image_ct)
        image_array = image_array.astype(np.float32)
        # [Z,Y,X] TO [X,Y,Z]
        image_array = image_array.transpose((2, 1, 0))
        image_array = torch.FloatTensor(image_array).unsqueeze(0)
        # get label tensor
        label = lable_loader(image, label, config.num_landmark)
        # 目标热力图label
        label = generate_heatmap_target1(list(config.heatmap_size), label, config.heatmap_sigma,
                                         scale=config.heatmap_scale, normalize=True)
        return image_array, label

    def __len__(self):
        return len(self.image_list)


def main():
    config = Config()
    image_test_list, label_test_list = get_file_list_gz(config.data_test_dir)
    transformer = 0  # 待加入
    # 加载训练集，测试集
    datasets_test = Datasets_test(image_test_list, label_test_list, transformer)
    datasets_test = DataLoader(datasets_test, batch_size=config.batch_size, shuffle=False)

    # 需要修改的参数
    model = Unet3d(in_channels=1, n_classes=5, n_channels=32).to(device)  # 网络
    # model = UNETR(in_channels=1, out_channels=5, img_size=(176, 176, 96), pos_embed='conv', norm_name='instance')
    model_load = '/DATA/lyh/code/3DUnet_origin/cache/model_saved/Unet_for_LSTM/checkpoint_epoch_0.pt'
    checkpoint = torch.load(model_load)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model_filename = '/DATA/lyh/code/3DUnet_origin/cache/model_saved/OH_lr_0.0001_e_500_128_Unet/model_2024-03-12_22-25-22.pt'
    # model = torch.load(model_filename)
    # model = model.to(device)
    print("\n------------------------------------------------start saving model--------------------------------------------\n")
    test(model, datasets_test)

    distance_count(43, model_path=save_dir, data_path=config.data_test_dir, device=device)

def test(model, testloader):
    #  --------------------------------------------------------------------------------------------------------------
    i = 43
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            print(y_pred.max())
            count = ((y_pred == y_pred.max()).sum().item())
            count_gt_0_5 = (y_pred.sigmoid() > 0.5).sum().item()
            count_eq_1 = (y_pred.sigmoid() == 1).sum().item()
            print(count_eq_1)
            print(count_gt_0_5)
            print(count)
            unique_elements, _ = torch.unique(y_pred, return_inverse=True)

            print(unique_elements)
            y_max_indices = find_max_indices1(y)
            y_pred_max_indices = find_max_indices(y_pred)

            torch.save(y_pred, f"{save_dir}predict_{i}.pth")
            print(f"saved predict_{i}.pth success!")
            i = i + 1
    # --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
def find_max_indices(tensor):
    max_indices_list = []

    for i in range(5):
        max_value = torch.max(tensor[:, i, :, :, :])
        max_indices = torch.nonzero(tensor[:, i, :, :, :] == max_value, as_tuple=False)
        max_indices_list.append(max_indices)
    max_indices_tensor = torch.cat(max_indices_list, dim=0)
    max_indices_tensor = max_indices_tensor[:, 1:]
    return max_indices_tensor

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
