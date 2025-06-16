import torch
import torch.nn as nn
class U_Net3D_encoder(nn.Module):
    def __init__(self, fin, fout):
        super(U_Net3D_encoder, self).__init__()
        ndf = 32
        # ~ conNum = 64
        self.Lconv1 = nn.Sequential(
            nn.Conv3d(fin, ndf, 3, 1, 1),
            nn.BatchNorm3d(ndf, track_running_stats=False),
            nn.ReLU(True),
        )
        self.Lconv2 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf, ndf * 2, 3, 1, 1),
            nn.Conv3d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm3d(ndf * 2, track_running_stats=False),
            nn.ReLU(True),
        )

        self.Lconv3 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf*2, ndf*4, 3, 1, 1),
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm3d(ndf*4,track_running_stats=False),
            nn.ReLU(True),

        )

        self.Lconv4 = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf*4, ndf*8, 3, 1, 1),
            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm3d(ndf*8,track_running_stats=False),
            nn.ReLU(True),
        )

        self.bottom_encoder = nn.Sequential(
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(ndf * 8, ndf * 16, 3, 1, 1),
            nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.BatchNorm3d(ndf * 16, track_running_stats=False),
            nn.ReLU(True),
            nn.AvgPool3d(2, 2),
        )

    def forward(self, x):
        x1 = self.Lconv1(x)
        x2 = self.Lconv2(x1)
        x3 = self.Lconv3(x2)
        x4 = self.Lconv4(x3)
        bottom = self.bottom_encoder(x4)
        # print(bottom.size())
        return bottom
        # return bottom
