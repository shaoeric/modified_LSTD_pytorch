import torch
import torch.nn as nn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]


def mask_vgg_layers(cfg=cfg):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)]

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(1, 1, kernel_size=3, padding=6, dilation=6, bias=False)
    conv7 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    layers += [pool5, conv6, conv7]
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            ones = torch.ones(size=(1, 1, *layer.kernel_size))
            layer.weight = torch.nn.Parameter(ones, requires_grad=False)
    return layers


# x = torch.Tensor(1, 3, 300, 300)
# mask = torch.zeros(size=(1, 1, 300, 300))
# mask[0, 0, 23, 125] = 1
# mask[0, 0, 54, 72] = 1
# mask[0, 0, 178, 72] = 1
# mask[0, 0, 18, 35] = 1
# mask[0, 0, 12, 51] = 1
# mask[0, 0, 1, 2] = 1
# origin_mask = mask.clone()
#
#
# for layer in mask_vgg_layers():
#     mask = layer(mask)
#     print(mask.sum())
# #     break