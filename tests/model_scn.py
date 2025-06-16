import torch
import torch.nn as nn
import torch.nn.functional as F


class SCN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SCN, self).__init__()
        num_filters_base = 64  # 设置基础的卷积核数量
        activation_conv = nn.LeakyReLU(negative_slope=0.1)  # 定义 leaky_relu 激活函数
        # padding = 1
        # 初始化热图层的卷积核参数
        # heatmap_layer_kernel_initializer = torch.empty(1)
        # nn.init.trunc_normal_(heatmap_layer_kernel_initializer, std=0.001, a=0.002, b=-0.002)
        # conv_heatmap = nn.Conv3d(num_filters_base, n_classes, kernel_size=7, stride=1, padding_mode='reflect')
        self.downsampled_factor = 4  # 下采样因子
        self.kernel_size = (self.downsampled_factor,) * 3
        self.SCN = nn.Sequential(
            nn.AvgPool3d(self.kernel_size),
            nn.Conv3d(in_channels, num_filters_base, kernel_size=7, stride=1, padding=3 ,padding_mode='reflect'),
            activation_conv,
            nn.Conv3d(num_filters_base, num_filters_base, kernel_size=7, stride=1, padding=3 ,padding_mode='reflect'),
            activation_conv,
            nn.Conv3d(num_filters_base, num_filters_base, kernel_size=7, stride=1, padding=3 ,padding_mode='reflect'),
            activation_conv,
            nn.Conv3d(num_filters_base, n_classes, kernel_size=7, stride=1, padding=3 ,padding_mode='reflect'),
            nn.Tanh(),
        )
        # SCN[-2].weight.data.fill_(heatmap_layer_kernel_initializer)
        nn.init.trunc_normal_(self.SCN[-2].weight, std=0.001, a=-0.002, b=0.002)

    def upsampled(self, x):
        return F.interpolate(x, scale_factor=self.downsampled_factor, mode='trilinear')
        # return F.interpolate(x, size=[128,128,56], mode='trilinear')
    # def init_weight(m):
    #     if isinstance(m, nn.Conv3d):
    #         nn.init.trunc_normal_(m.weight, std=0.001, a=0.002, b=-0.002)


    def forward(self, x):
        local_heatmaps = x
        x = self.SCN(x)
        x = self.upsampled(x)
        return x * local_heatmaps


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
    def __init__(self, in_channels, n_classes, n_channels):
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
        self.SCN = SCN(n_classes, n_classes)

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

        mask = self.Out(mask)
        mask_final = self.SCN(mask)
        # print('mask.shape:', mask.shape)
        return mask ,mask_final
