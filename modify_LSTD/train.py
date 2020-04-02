import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from layers.modules import MultiBoxLoss, ClassifierLoss
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import os
import config
from utils.plot import *

from utils.auguments import SSDAugmentation
from torch.utils.data import DataLoader
from data import detection_collate
from data.voc0712 import VOCDetection

from models.lstd_source_bd import build_ssd

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if torch.cuda.is_available():
    if config.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(config.save_folder):
    os.mkdir(config.save_folder)


def train():
    # 数据集
    dataset = VOCDetection(config.VOC_ROOT, transform=SSDAugmentation(config.voc['min_dim']), mask=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=detection_collate, pin_memory=True, shuffle=True)
    batch_iterator = iter(dataloader)

    # 模型
    lstd = build_ssd('train', config.voc['min_dim'], config.voc['num_classes'])
    net = lstd

    print("loading base network...")
    net.vgg.load_state_dict(torch.load(os.path.join(config.pretrained_folder, config.basenet)))

    if config.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    rpn_loss = MultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, config.cuda)
    mask_loss = nn.CrossEntropyLoss()
    conf_loss = ClassifierLoss()

    net.train()
    step_index = 0  # 用于lr的调节
    for iteration in range(config.voc['max_iter']):
        if iteration in config.voc['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, config.gamma, step_index)

        try:
            images, targets, masks = next(batch_iterator)
        except StopIteration as e:
            batch_iterator = iter(dataloader)
            images, targets, masks = next(batch_iterator)

        if config.cuda:
            images, masks = images.cuda(), masks.cuda
            targets = [ann.cuda() for ann in targets]

        confidence, roi, rpn_out, mask_out = net(images)
        loss_c = conf_loss.forward(roi, targets, confidence)
        break
        # optimizer.zero_grad()
        # loss_rpn_out = rpn_loss(rpn_out, targets)

        # 可能需要把target转换成21列，，然后用conf_loss, 这部分的loss module要参考multiboxloss，因为num_roi 与target的个数不匹配

        # 需要把target的部分，转换成01形式，有object即为1，否则为0，然后使用multiboxloss


def adjust_learning_rate(optimizer, gamma, step):
    lr = config.lr *(gamma**step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()










