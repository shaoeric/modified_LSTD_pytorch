import torch
import torch.nn as nn



class MaskGenerate(nn.Module):
    def __init__(self, channel=3, mid_channel=64, phase='train', thresh=0.5):
        """
        背景抑制，是对图像进行编解码，得到其mask层
        :param channel:
        :param mid_channel:
        :param phase: str: "train" or "test"
        :param thresh: float, 当phase为test时，对重建结果进行分割，大于阈值则为1，小于阈值为0
        """
        super(MaskGenerate, self).__init__()
        self.phase = phase
        self.thresh = thresh

        self.encoder = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, mid_channel, kernel_size=3, dilation=2),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(mid_channel, 32, kernel_size=3, dilation=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)  # [batchsize, 64, 144, 144]
        x = self.decoder(x)  # [batchsize, 1, 300, 300]
        # if self.phase == 'test':
        x[x.ge(self.thresh)] = 1
        x[x.lt(self.thresh)] = 0
        return x


if __name__ == '__main__':
    img = torch.Tensor(1, 3, 38, 38)
    bd = MaskGenerate(phase="test")
    out = bd(img)
    print(out.shape)
    a = (out == 0).sum()
    print(a)