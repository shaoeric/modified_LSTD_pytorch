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

from models.lstd_source import build_ssd as build_source
from models.lstd_target import build_ssd as build_target

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
    source = build_source('test', config.voc['min_dim'])
    target_net = build_target('train', config.voc['min_dim'], True)
    target_net.mask_generator.apply(weights_init)
    print(source)
    print(target_net)

    print("loading base network...")
    source_weights = torch.load(os.path.join(config.save_folder, 'lstd_source12500.pth'), map_location=config.device)
    source.load_state_dict(source_weights)

    for k, v in target_net.named_parameters():
        if k in source_weights.keys():
            v.data = source_weights[k].data  # 直接加载预训练参数
        else:
            try:
                if k.find("weight") >= 0:
                    nn.init.xavier_normal_(v.data)  # 没有预训练，则使用xavier初始化
                else:
                    nn.init.constant_(v.data, 0)  # bias 初始化为0
            except:  # maskgenerator层 存在channel小于2的，会报错 所以提前初始化，这里捕获跳过
                pass

    if config.cuda:
        # net = torch.nn.DataParallel(net, [0])
        # cudnn.benchmark = True
        source = source.cuda(device=config.device)
        target_net = target_net.cuda(device=config.device)

    # torch.nn.utils.clip_grad_norm(parameters=net.module.classifier.parameters(), max_norm=10, norm_type=2)
    optimizer = optim.Adam(
        [{'params': target_net.vgg.parameters()},
        {'params': target_net.extras.parameters()},
        {'params': target_net.loc.parameters()},
        {'params': target_net.conf.parameters()},
        {'params': target_net.mask_feature_map.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay},
        {'params': target_net.mask_generator.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay}], lr=config.lr/100, weight_decay=config.weight_decay
    )

    rpn_loss_func = MultiBoxLoss(2, config.rpn_iou_label_thresh, True, 0, True, 3, 0.5, False, config.cuda) # 判断是否为物体，所以
    # 只有2类
    mask_loss_func = nn.BCELoss()
    conf_loss_func = ClassifierLoss(num_classes=config.source_num_classes, focal_loss=False, iou_thresh=config.classifier_iou_label_thresh)
    bd_regulation_func = MaskBlock(is_bd=True)
    source.eval()
    target_net.train()

    step_index = 0  # 用于lr的调节
    train_classifier = False
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

        if not train_classifier:  # 只训练rpn
            rpn_out, mask_out = target_net(images, train_classifier)
            loss_loc, loss_obj = rpn_loss_func.forward(rpn_out, targets)  # objectness and loc loss
            loss_mask = mask_loss_func(mask_out.view(mask_out.size(0), -1), masks.view(masks.size(0), -1))
            # bd_regulation = bd_regulation_func.forward(bd_feature, masks)
            # # l1正则数值过大，达到4000多，l2正则比较平滑，调试过程中遇到的最大为80
            # bd_regulation = torch.sqrt(torch.sum(bd_regulation**2)) / torch.mul(*bd_regulation.shape[:2])
            loss = loss_loc + loss_obj + loss_mask #+ loss_c # #+ bd_regulation
            # loss = loss_obj + loss_c # #+ bd_regulation
            if iteration % 10 == 0:
                print('iter: {} || loss:{:.4f} || loss_loc:{:.4f} || loss_obj:{:.4f} || loss_mask'.format(repr(iteration), loss, loss_loc, loss_obj, loss_mask))

            if loss <= 5.5:
                rpn_loss_early_stop += 1

            if iteration >= config.rpn_train_max_iteration or rpn_loss_early_stop >= 50:  # 开始训练分类器，调整模式，固定rpn的参数不参与训练
                train_classifier = True
                optimizer = optim.Adam([
                    {'params': target_net.roi_pool.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay},
                    {'params': target_net.classifier.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay},
                    {'params': target_net.classifier_target.parameters(), 'lr':config.lr, 'weight_decay': config.weight_decay},
                ])
                iteration = 0
                print("开始训练分类器, early_stop=", rpn_loss_early_stop)

        else:
            confidence, roi, rpn_out, keep_count = target_net.forward(images, train_classifier)
            if keep_count.sum() == 0:  # 没有得到正样本
                print("no positive samples")
                continue

            result, num = conf_loss_func.forward(roi, targets, confidence)  # classification loss
            # 没有得到label的情况
            if not result:
                print("no positive assigned labels")
                continue
            else:
                loss = result
                if iteration % 10 == 0:
                    print('iter: {} || loss:{:.4f}, pos:{}'.format(repr(iteration), loss, num))

        loss.backward()
        optimizer.step()

        if iteration != 0 and iteration % 500 == 0:
            mode = 'whole' if train_classifier else 'rpn'
            name = 'weights/lstd_source_' + mode + repr(iteration) + '.pth'
            print('Saving state:', name)
            torch.save(source.state_dict(), name)

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










