import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from layers.modules import MultiBoxLoss, ClassifierLoss
from layers.functions import MaskBlock
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
    lstd = build_ssd('train', config.voc['min_dim'])
    net = lstd

    print("loading base network...")
    net.vgg.load_state_dict(torch.load(os.path.join(config.pretrained_folder, config.basenet)))

    if config.cuda:
        net = torch.nn.DataParallel(net, [0])
        cudnn.benchmark = True
        net = net.cuda()

    torch.nn.utils.clip_grad_norm(parameters=net.parameters(), max_norm=10, norm_type=2)
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    rpn_loss_func = MultiBoxLoss(2, 0.3, True, 0, True, 3, 0.5, False, config.cuda) # 判断是否为物体，所以
    # 只有2类
    mask_loss_func = nn.BCELoss()
    conf_loss_func = ClassifierLoss(num_class=config.num_classes, focal_loss=True)
    bd_regulation_func = MaskBlock(is_bd=True)
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

        masks = masks.float()
        if config.cuda:
            images, masks = images.cuda(), masks.cuda()
            targets = [ann.cuda() for ann in targets]

        confidence, roi, rpn_out = net(images)

        loss_loc, loss_obj = rpn_loss_func.forward(rpn_out, targets)  # objectness and loc loss
        loss_c = conf_loss_func.forward(roi, targets, confidence)  # classification loss
        # loss_mask = mask_loss_func(mask_out.view(mask_out.size(0), -1), masks.view(masks.size(0), -1))

        # bd_regulation = bd_regulation_func.forward(bd_feature, masks)
        # # l1正则数值过大，达到4000多，l2正则比较平滑，调试过程中遇到的最大为80
        # bd_regulation = torch.sqrt(torch.sum(bd_regulation**2)) / torch.mul(*bd_regulation.shape[:2])
        loss = loss_loc + loss_obj + loss_c # #+ bd_regulation
        # loss = loss_obj + loss_c # #+ bd_regulation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if iteration % 1 == 0:
            print('iter: {} || loss:{:.4f} || loss_loc:{:.4f} || loss_obj:{:.4f} || loss_c:{:.4f} '.format(repr(iteration), loss, loss_loc, loss_obj, loss_c))

        if iteration != 0 and iteration % 3000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'weights/lstd_source' +
                       repr(iteration) + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    lr = config.lr *(gamma**step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()










