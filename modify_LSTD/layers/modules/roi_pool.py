import torch
import torch.nn as nn

torch.set_printoptions(threshold=float('inf'))
class RoIPooling(nn.Module):
    def __init__(self, pooled_size=7, img_size=300, conved_channels=256, conved_size=7):
        super(RoIPooling, self).__init__()
        self.pooled_size = pooled_size
        self.img_size = img_size
        self.conved_channels = conved_channels
        self.conved_size = conved_size
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.pooled_size, self.pooled_size))
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, self.conved_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conved_channels),
            nn.ReLU(True),
            nn.Conv2d(self.conved_channels, self.conved_channels//8, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.conved_channels//8),
            nn.ReLU(True)
        )

    def forward(self, roi_proposal, features):
        """
        要根据原图宽高和300x300比例转换.但转换完后，原图的尺寸被约掉了，所以还是乘以300，
        为了映射到特征图上，还要除以16
        :param roi_proposal:  rpn经过转换后的roi区域 [batchsize,1, num_rois, 5]  5:[图像id，xmin,ymin,xmax,ymax]
        :param features:  vgg提取的特征图，[batchsize, 1024, 19, 19]
        :return: output: [batchsize, num_rois, 256, 2, 2]
        """
        rois = roi_proposal.clone()
        batch = rois.size(0)
        num_rois = rois.size(2)  # top_k

        rois[:, :, :, 1:] = rois[:, :, :, 1:] * self.img_size / 16
        rois = rois.long()[:, :, :, 1:]

        effective_rois_features = torch.zeros(size=(features.size(0), num_rois, self.conved_channels//8, self.conved_size, self.conved_size)).type(features.type()) # 有效roi的roi_pooling特征图
        keep_count = torch.zeros(size=(batch,)).long()  # 用来记录每一张图片 有多少个roi是有效的

        effective_rois = torch.zeros(size=rois.shape)
        for i in range(batch):
            n = 0
            for r in range(num_rois):
                roi = rois[i, 0, r]
                x_min, y_min, x_max, y_max = roi.clamp(min=0, max=self.img_size / 16)
                if x_max <= x_min or y_max <= y_min:
                    continue

                effective_rois[i, 0, n] = roi.float() * 16 / self.img_size  # 注意要加float，要不然全都是零
                out = features.narrow(0, i, 1)[..., y_min: y_max+1, x_min: x_max+1]
                out = self.pool(out)
                out = self.conv(out)
                effective_rois_features[i, n, ...] = out
                n += 1
            keep_count[i] = n
        return effective_rois, effective_rois_features, keep_count
