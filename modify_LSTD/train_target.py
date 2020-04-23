import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from layers.modules import MultiBoxLoss, ClassifierLoss, AdaptLoss
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
from data.voc0712 import CustomDataset

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
    dataset = CustomDataset(config.target_VOC_ROOT, config.target_VOC_Annotations, transform=SSDAugmentation(config.voc['min_dim']), mask=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=detection_collate, pin_memory=True, shuffle=True)
    batch_iterator = iter(dataloader)

    # 模型
    source = build_source('transfer', config.voc['min_dim'])
    target_net = build_target('train', config.voc['min_dim'], True)
    target_net.mask_generator.apply(weights_init)
    # print(source)
    # print(target_net)

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

    # torch.nn.utils.clip_grad_norm(parameters=net.module.classifier_tk.parameters(), max_norm=10, norm_type=2)
    optimizer = optim.Adam(
        [{'params': target_net.vgg.parameters()},
        {'params': target_net.extras.parameters()},
        {'params': target_net.loc.parameters()},
        {'params': target_net.conf.parameters()},
        {'params': target_net.mask_feature_map.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay},
        {'params': target_net.mask_generator.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay}], lr=config.lr/10, weight_decay=config.weight_decay
    )

    rpn_loss_func = MultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5, False, config.cuda) # 判断是否为物体，所以
    # 只有2类
    mask_loss_func = nn.BCELoss()
    conf_loss_func = ClassifierLoss(num_classes=config.target_num_classes, focal_loss=False, iou_thresh=config.classifier_iou_label_thresh)
    bd_regulation_func = MaskBlock(is_bd=True)
    tk_regulation_func = AdaptLoss()
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
            loss_mask = mask_loss_func(mask_out.view(mask_out.size(0), -1), masks.view(masks.size(0), -1)) * 10
            # bd_regulation = bd_regulation_func.forward(mask_out, masks)
            # # l1正则数值过大，达到4000多，l2正则比较平滑，调试过程中遇到的最大为80
            # bd_regulation = torch.sqrt(torch.sum(bd_regulation**2)) / torch.mul(*bd_regulation.shape[:2])
            loss = loss_loc + loss_obj + loss_mask #+ loss_c # #+ bd_regulation
            # loss = loss_obj + loss_c # #+ bd_regulation
            if iteration % 10 == 0:
                print('iter: {} || loss:{:.4f} || loss_loc:{:.4f} || loss_obj:{:.4f} || loss_mask:{:.4f}'.format(repr(iteration), loss, loss_loc, loss_obj, loss_mask))

            if loss <= 5.5:
                rpn_loss_early_stop += 1

            if iteration >= 400 or rpn_loss_early_stop >= 50:  # 开始训练分类器，调整模式，固定rpn的参数不参与训练
                train_classifier = True
                optimizer = optim.Adam([
                    {'params': target_net.roi_pool.parameters(), 'lr': config.lr, 'weight_decay': config.weight_decay},
                    {'params': target_net.classifier_target.parameters(), 'lr':config.lr, 'weight_decay': config.weight_decay},
                    {'params': target_net.classifier.parameters(), 'lr': config.lr / 10, 'weight_decay': config.weight_decay},
                    {'params': source.roi_pool.parameters(), 'lr': config.lr / 10, 'weight_decay': config.weight_decay},
                    {'params': source.classifier.parameters(), 'lr': config.lr / 10, 'weight_decay': config.weight_decay},
                ])
                iteration = 0
                print("开始训练分类器, early_stop=", rpn_loss_early_stop)
                name = 'weights/lstd_target_' + "rpn" + repr(iteration) + '.pth'
                print('Saving state:', name)
                torch.save(target_net.state_dict(), name)
        else:
            #### Todo target的分类器需要与truth的交并比 做分类，也要和sourcenet的roi做交并比做分类，因为target做了微调，box可能出现变化
            confidence_tk, rois, target_confidence, keep_count, roi_origin, feature = target_net.forward(images, train_classifier)

            if keep_count.sum() == 0:  # 没有得到正样本
                print("no positive samples")
                continue
            result_truth, num = conf_loss_func.forward(rois, targets, target_confidence)  # classification loss
            # 没有得到label的情况
            if not result_truth:
                print("no positive assigned labels")
                continue
            else:
                # source网络只需要留下roipooling和classifier
                source_roi, source_roi_out, source_keep_count = source.roi_pool.forward(roi_origin, feature)
                source_conf = source.classifier.forward(source_roi_out, source_keep_count).to(config.device)
                source_roi = source_roi[:, :, :source_conf.size(1), :]

                tk_reg = tk_regulation_func.forward(rois, confidence_tk, source_roi, source_conf)
                loss = result_truth + tk_reg
                if iteration % 1 == 0:
                    print('iter: {} || loss:{:.4f} || loss_conf:{:.4f} || tk:{:.4f}'.format(repr(iteration), loss, result_truth, tk_reg))

        loss.backward()
        optimizer.step()

        if iteration != 0 and iteration % 500 == 0:
            mode = 'whole' if train_classifier else 'rpn'
            name = 'weights/lstd_target_' + mode + repr(iteration) + '.pth'
            print('Saving state:', name)
            torch.save(target_net.state_dict(), name)

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










