import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.functions import PriorBox, Detect, MaskBlock
from layers.modules import L2Norm, MultiBoxLoss, RoIPooling, Classifier, MaskGenerate, ConvFeatureCompress
import config
import os
from layers.modules import Post_rois


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

    def __init__(self, phase, base, extras, head, num_classes, target):
        super(SSD, self).__init__()
        self.phase = phase
        self.target = target
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

        # 目标域蒙版生成部分
        if self.target:
            self.mask_feature_map = nn.ModuleList([
                ConvFeatureCompress(),  # 从512x38x38的特征图映射到3x38x38特征图
                nn.Conv2d(1, 1, kernel_size=2, stride=2)    # 1x38x38 => 1x19x19
            ]).to(config.device)
            self.mask_generator = MaskGenerate(4, 64).to(config.device)
            # self.mask_block = MaskBlock()

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.post_roi = Post_rois(2, 0, config.top_k, config.conf_thresh, config.rpn_nms_thresh)
        self.detect = Detect(config.target_num_classes, 0, config.selected_proposal, config.conf_thresh, config.nms_thresh)
        # target net module
        self.roi_pool = RoIPooling(pooled_size=config.pooled_size, img_size=self.size, conved_size=config.pooled_size, conved_channels=config.conved_channel)
        self.classifier = Classifier(num_classes=config.source_num_classes)
        self.classifier_target = Classifier(num_classes=config.target_num_classes)
        if use_cuda:
            self.post_roi = self.post_roi.cuda(device=config.device)
            self.roi_pool = self.roi_pool.cuda(device=config.device)
            self.classifier = self.classifier.cuda(device=config.device)
            self.classifier_target = self.classifier_target.cuda(device=config.device)
            self.priors = self.priors.cuda(device=config.device)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, train_classifier=False):
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
                    1: confidence layers, Shape: [batch*num_priors,source_num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        img = x.clone()
        # self.train_classifier = train_classifier if train_classifier else self.train_classifier

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        bd_feature = x.clone()
        # 生成蒙版
        feature_map = self.mask_feature_map[0](bd_feature)
        mask_38 = self.mask_generator(feature_map)  # [1, 1, 38, 38]  用来训练 优化loss

        s = self.L2Norm(x)
        sources.append(s)   # [1, 512, 38, 38]

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

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

        # 训练阶段 只训练rpn， 这部分可能也要分开，rpn部分只需要finetune就好 但mask部分要训练多一些
        if not train_classifier:
            return rpn_output, mask_38

        rois = self.post_roi.forward(*rpn_output)[:, 1:, :, :].to(config.device)
        roi_origin = rois.clone()
        rois, roi_out, keep_count = self.roi_pool(rois, sources[1])

        # 分类输出（带背景）
        if self.phase == 'train':
            confidence_tk = self.classifier(roi_out, keep_count).to(config.device)  # [batchsize, top_k, source_num_classes+1]

        # 还有一个target的confidence
        target_confidence = self.classifier_target(roi_out, keep_count).to(config.device)
        rois = rois[:, :, :target_confidence.size(1), :]  # [batch, 1, 100, 4]

        if self.phase == "train":
            return confidence_tk, rois, target_confidence, keep_count, roi_origin, sources[1], rpn_output, mask_38

        elif self.phase == "detect":
            confidence = self.softmax(target_confidence.view(conf.size(0), -1, config.target_num_classes))
            return self.detect.forward(confidence, rois), mask_38
        else:
            target_confidence = self.softmax(target_confidence.view(conf.size(0), -1, config.target_num_classes))
            return target_confidence, rois,   #mask_38, None

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


def build_ssd(phase, size=300, target=True):  # base ssd只检测objectness
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], base_classes=2)
    return SSD(phase, base_, extras_, head_, 2, target)


if __name__ == '__main__':
    net = build_ssd("train").cuda()
    # net.eval()
    img = torch.rand(size=(1, 3, 300, 300)).cuda()
    out = net(img)

    net.print_params_num()