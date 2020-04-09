import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.functions import PriorBox, Detect, MaskBlock
from layers.modules import L2Norm, MultiBoxLoss, RoIPooling, Classifier, MaskGenerate
import config
import os
from layers.functions.post_rois import Post_rois
from copy import deepcopy

use_cuda = torch.cuda.is_available()

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, extras_lstd, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes  # objectness 所以是2
        self.cfg = config.voc
        self.priorbox = PriorBox(self.cfg)

        with torch.no_grad():
            self.priors = self.priorbox.forward()

        self.size = 300
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        # self.mask_feature_map = nn.ModuleList([
        #     nn.Conv2d(512, 3, kernel_size=3, padding=1, bias=False),  # 从512x38x38的特征图映射到3x38x38特征图
        #     nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)    # 1x38x38 => 1x19x19
        # ])
        # self.mask_generator = MaskGenerate(3, 64, self.phase, thresh=config.mask_thresh)
        # self.mask_block = MaskBlock()

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        # faster rcnn part
        self.post_rois = Post_rois(self.num_classes, 0, config.top_k, config.conf_thresh, config.nms_thresh) # 背景+前景只有2类
        self.roi_pool = RoIPooling(pooled_size=config.pooled_size, img_size=self.size, conved_size=config.pooled_size, conved_channels=config.conved_channel)
        self.classifier = Classifier(num_classes=config.num_classes)
        if use_cuda:
            # self.mask_generator = self.mask_generator.cuda()
            self.roi_pool = self.roi_pool.cuda()
            self.classifier = self.classifier.cuda()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        img = x.clone()

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        bd_feature = x.clone()
        # 生成蒙版
        # mask_38 = None
        # if self.phase == 'train':
        # feature_map = self.mask_feature_map[0](x)
        # mask_38 = self.mask_generator(feature_map)  # [1, 1, 38, 38]  用来训练 优化loss
        # mask_19 = self.mask_feature_map[1](mask_38)  # [1, 1, 19, 19]  用来掩盖下一层的背景

        s = self.L2Norm(x)
        sources.append(s)   # [1, 512, 38, 38]

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            # if k == 23 and self.phase == 'train':
                # x = self.mask_block.forward(x, mask_19)  # 蒙版掩膜

        sources.append(x)   # [1, 1024, 19, 19]

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # SSD prediction
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # rpn 和rois是相对于原图的box坐标比例值，而不是相对于300x300的
        rpn_output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),  # [batch, 8732, 2]
            self.priors
        )
        with torch.no_grad():
            rois = self.post_rois.forward(img, *rpn_output)  # rois.size (batch,1, top_k,5)  scaled[0, 1]
        #  faster rcnn roi pooling，
        roi_out = self.roi_pool(rois, sources[1])  # [batch, top_k, 128, 7, 7]

        # 分类输出（带背景）
        confidence = self.classifier(roi_out)  # [batchsize, top_k, num_classes+1]

        if self.phase == "train":
            return confidence, rois, rpn_output, #mask_38, bd_feature
        else:
            confidence = self.softmax(confidence.view(conf.size(0),-1, config.num_classes))
            return confidence, rois, #mask_38, None

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def print_params_num(self):
        total = sum([param.nelement() for param in self.parameters()])
        print("number of the total parameters:{}M".format(total/1e6))


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def add_lstd_extras(i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    conv_add1 = nn.Conv2d(in_channels, 256,
                          kernel_size=3, stride=1, padding=1)

    in_channels = 256
    batchnorm_add1 = nn.BatchNorm2d(in_channels)
    conv_add2 = nn.Conv2d(in_channels, 256,
                          kernel_size=3, stride=2, padding=1)

    batchnorm_add2 = nn.BatchNorm2d(in_channels)
    # bbox_score_voc = nn.Linear(256, 21)

    layers += [conv_add1, batchnorm_add1, nn.ReLU(inplace=True), conv_add2, batchnorm_add2, nn.ReLU(inplace=True)]
    return layers


def multibox(vgg, extra_layers, cfg, base_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * base_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * base_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.ReLU(True), nn.BatchNorm2d(cfg[k])]
    conf_layers += [nn.ReLU(True), nn.BatchNorm2d(cfg[k])]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, base_classes=2):  # base ssd只检测objectness
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], base_classes=base_classes)
    extras_lstd_ = add_lstd_extras(1024)
    # return SSD(phase, base_, extras_, head_, num_classes)
    return SSD(phase, base_, extras_, head_, extras_lstd_, base_classes)


if __name__ == '__main__':
    net = build_ssd("train", base_classes=2).cuda()
    # net.eval()
    img = torch.rand(size=(1, 3, 300, 300)).cuda()
    out = net(img)

    # net.print_params_num()