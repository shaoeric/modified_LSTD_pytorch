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

from models.lstd_source import build_ssd

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

print()
def train():
    # 数据集
    dataset = VOCDetection(config.VOC_ROOT, transform=SSDAugmentation(config.voc['min_dim']), mask=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=detection_collate, pin_memory=True, shuffle=True)
    batch_iterator = iter(dataloader)

    # 模型
    lstd = build_ssd('train', config.voc['min_dim'])
    net = lstd

    print(net)

    print("loading base network...")
    net.vgg.load_state_dict(torch.load(os.path.join(config.pretrained_folder, config.basenet)))
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.roi_pool.apply(weights_init)
    net.classifier.apply(weights_init)


    if config.cuda:
        # net = torch.nn.DataParallel(net, [0])
        # cudnn.benchmark = True
        net = net.cuda(device=config.device)

    # torch.nn.utils.clip_grad_norm(parameters=net.module.classifier.parameters(), max_norm=10, norm_type=2)
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rpn_loss_func = MultiBoxLoss(2, config.rpn_iou_label_thresh, True, 0, True, 3, 0.5, False, config.cuda) # 判断是否为物体，所以
    # 只有2类
    mask_loss_func = nn.BCELoss()
    conf_loss_func = ClassifierLoss(num_classes=config.source_num_classes, focal_loss=False, iou_thresh=config.classifier_iou_label_thresh)
    bd_regulation_func = MaskBlock(is_bd=True)
    net.train()

    step_index = 0  # 用于lr的调节
    rpn_loss_early_stop = 0
    iteration = 0
    while iteration <= config.voc['max_iter']:
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
            images, masks = images.cuda(device=config.device), masks.cuda(config.device)
            targets = [ann.cuda(device=config.device) for ann in targets]

        optimizer.zero_grad()


        confidence, roi, keep_count, rpn_out = net.forward(images, True)
        if keep_count.sum() == 0:  # 没有得到正样本
            print("no positive samples")
            continue
        result, num = conf_loss_func.forward(roi, targets, confidence)  # classification loss
        # 没有得到label的情况
        if not result:
            print("no positive assigned labels")
            continue
        else:
            loss_loc, loss_obj = rpn_loss_func.forward(rpn_out, targets)
            loss = loss_obj + loss_loc + result
            if iteration % 10 == 0:
                print('iter: {} || loss:{:.4f} || loss_loc:{:.4f}|| loss_obj:{:.4f} || loss_conf:{:.4f}|| pos:{}'.format(repr(iteration), loss, loss_loc, loss_obj, result, num))


        loss.backward()
        optimizer.step()

        if iteration != 0 and iteration % 500 == 0:
            name = 'weights/lstd_source_' + repr(iteration) + '.pth'
            print('Saving state:', name)
            torch.save(net.state_dict(), name)

        iteration += 1


def adjust_learning_rate(optimizer, gamma, step):
    lr = config.lr *(gamma**step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)

if __name__ == '__main__':
    train()










