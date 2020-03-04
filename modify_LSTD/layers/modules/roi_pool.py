import torch
import torch.nn as nn


class RoIPooling(nn.Module):
    def __init__(self, pooled_size=5, img_size=300, conved_channels=256, conved_size=2):
        super(RoIPooling, self).__init__()
        self.pooled_size = pooled_size
        self.img_size = img_size
        self.conved_channels = conved_channels
        self.conved_size = conved_size
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.pooled_size, self.pooled_size))
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

    def forward(self, rois, features):
        """
        要根据原图宽高和300x300比例转换.但转换完后，原图的尺寸被约掉了，所以还是乘以300，
        为了映射到特征图上，还要除以16
        :param rois:  rpn经过转换后的roi区域 [batchsize,1, num_rois, 5]  5:[图像id，xmin,ymin,xmax,ymax]
        :param features:  vgg提取的特征图，[batchsize, 1024, 19, 19]
        :return: output: [batchsize, num_rois, 256, 2, 2]
        """
        num_rois = rois.size(2)
        rois = rois.requires_grad_(False)
        rois = rois.reshape(-1,  5)
        rois[:, 1:] = rois[:,  1:] * self.img_size / 16
        rois = rois.long()

        output = torch.zeros(size=(features.size(0), num_rois, self.conved_channels, self.conved_size, self.conved_size)).type(features.type())

        for i in range(num_rois):
            roi = rois[i]
            img_idx = roi[0]
            x_min, y_min, x_max, y_max = roi[1:].clamp(min=1, max=18)
            x_min = (x_min - 1) if x_max == x_min else x_min
            y_min = (y_min - 1) if y_max == y_min else y_min
            out = features.narrow(0, img_idx, 1)[..., y_min: y_max+1, x_min: x_max+1]

            out = self.pool(out)  # [1, 1024, 5, 5]
            out = self.conv(out)  # [1, 256, 2, 2]
            output[img_idx, i, ...] = out

        return output  # [2, 2, 256, 2, 2]
