import torch
import torch.nn as nn
import torch.nn.functional as F
import network_layer.U_net3d as unet
from utils.Utils import get_global_feature
# from monai.networks.nets import unet


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=2):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNormal3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)
# 增加dropout层
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, num_groups=2, dropout_rate=0.0):
#         super(DoubleConv, self).__init__()
#         layers = [
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout3d(dropout_rate),  # 添加Dropout层
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.ReLU(inplace=True)
#         ]
#         self.double_conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):

        super(Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels, dropout_rate=0.0):
        super(Unet3d, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)

        self.Out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)

        mask_final = self.Out(mask)
        # print('mask.shape:', mask.shape)
        return mask, mask_final


class graph_attention(nn.Module):
    def __init__(self,feature_size,device):
        super(graph_attention,self).__init__()
        c = feature_size
        self.c = c
        self.device = device
        self.i  = nn.Sequential(
            nn.Linear(c+3,c//2)
        )
        self.j = nn.Sequential(
            nn.Linear(c+3, c//2),
        )
        self.k = nn.Sequential(
            nn.Linear(c+3, c//2),
        )
        self.restore = nn.Sequential(
            nn.Linear(c//2, c),
        )

    def forward(self,landmarks,features):
        input_size = features.size()
        # (landmarks,channels+xyz:3)
        features_concat = torch.cat((landmarks,features),dim=2).squeeze()
        fi = self.i(features_concat)
        fj = self.j(features_concat).permute(1, 0)
        fk = self.k(features_concat)
        attention_ij = torch.sigmoid(torch.matmul(fi,fj))
        attention_sum = torch.sum(attention_ij,1).view(-1,1)
        attention_ij = attention_ij/attention_sum
        features = features + self.restore(torch.matmul(attention_ij,fk))

        return features.view(input_size)




        # input_size = feature_size()
        # features_concat = torch.cat((ROIs,features),dim=2).squeeze()

class fine_gat(nn.Module):
    def __init__(self,config):
        super(fine_gat,self).__init__()
        self.landmarkNum = config.num_landmark
        self.device = config.device
        # self.origin_image_size = config.origin_image_size
        self.config = config
        self.feature_size = config.feature_size

        self.enconder = unet.U_Net3D_encoder(1,self.feature_size)
        self.decoders_offset_x = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + self.feature_size, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_y = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + self.feature_size, 1, 0, groups=self.landmarkNum)
        self.decoders_offset_z = nn.Conv1d(self.landmarkNum, self.landmarkNum, 512 + self.feature_size, 1, 0, groups=self.landmarkNum)

        self.attention_gate_head = nn.Conv1d(self.landmarkNum, self.landmarkNum, 256, 1, 0, groups=self.landmarkNum)
        self.graph_attention = graph_attention(self.feature_size, self.device)

    def forward(self,crop_data, coarse_landmarks,coarse_feature):
        #     A[原始输入] --> B[裁剪ROI区域]
        # data_crop = get_patch(image_info["origin"],coordinate,image_info["image_size"],device)
        #     B --> C[局部特征提取]
        [h,w,l] = self.config.image_size

        cropedtems = torch.cat([crop_data[i].to(self.device) for i in range(len(crop_data))], dim=0)
        features = self.enconder(cropedtems).squeeze().unsqueeze(0)

        ROIS = coarse_landmarks / torch.tensor([h, w, l], dtype=coarse_landmarks.dtype, device=coarse_landmarks.device)

        # ROIS (1,24,3)  coarse feature [1,48,96,96,96] landmarkNum 24
        global_feature = get_global_feature(ROIS, coarse_feature, self.landmarkNum)
        #     A --> D[生成全局特征]
        global_feature = self.graph_attention(ROIS, global_feature)

        #     D --> E[图注意力增强]
        #     C & E --> F[特征拼接融合]
        features = torch.cat((features, global_feature), dim=2)
        x, y, z = self.decoders_offset_x(features), self.decoders_offset_y(features), self.decoders_offset_z(features)
        size_inv = torch.tensor(
            [h, w, l],
            dtype=features.dtype,
            device=self.device
        ).view(1, 1, 3)

        off = torch.cat([x, y, z], dim=2)
        # 使用tanh将输出限制在[-1, 1]，然后乘以5得到[-5, 5]
        off = torch.tanh(off) * 10
        predict = coarse_landmarks + off
        return predict

