import torch
import torch.nn as nn
import config


class ConvFeatureCompress(nn.Module):
    def __init__(self):
        super(ConvFeatureCompress, self).__init__()

    def forward(self, x):
        min_map, _ = torch.min(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        var_map = torch.var(x, dim=1, keepdim=True)
        mean_map = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([min_map, max_map, var_map, mean_map], dim=1)



class MaskGenerate(nn.Module):
    def __init__(self, channel=3, mid_channel=64):
        """
        背景抑制，是对图像进行编解码，得到其mask层
        :param channel:
        :param mid_channel:
        :param thresh: float, 当phase为test时，对重建结果进行分割，大于阈值则为1，小于阈值为0
        """
        super(MaskGenerate, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, mid_channel, kernel_size=3, dilation=2),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.1, True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channel, 32, kernel_size=3, dilation=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        value = torch.tensor(0.).to(device=x.device)
        x = torch.where(torch.isnan(x), value, x)
        x = self.encoder(x)  # [batchsize, 64, 13, 13]
        x = self.decoder(x)  # [batchsize, 1, 38, 38]
        return x


if __name__ == '__main__':
    while True:
        img = torch.randn(size=(1, 3, 38, 38))
        bd = MaskGenerate()
        out = bd(img)
        bce = nn.BCELoss()
        try:
            loss = bce(out.view(1, -1), img[0,0, ...].view(1, -1))
            loss.backward()
            print(img.max(), img.min(), out.max(), out.min(), loss)
        except:
            print("except")
            print(out.max(), out.min())
            break
    # import numpy as np
    # x = torch.tensor(np.nan)
    # x = torch.where(torch.isnan(x), torch.tensor(0.), x)
    # print(x)