
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import assign_label_for_rois
import config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=21, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retinanet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha) == num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha < 1   # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class ClassifierLoss(nn.Module):
    def __init__(self, num_class=21, focal_loss=False):
        super(ClassifierLoss, self).__init__()
        self.focal_loss = focal_loss
        if focal_loss:
            self.loss_func = FocalLoss(num_classes=num_class, size_average=False)
        else:
            self.loss_func = F.cross_entropy

    def forward(self, rois, targets, prediction):
        """

        :param rois: scaled tensor [batchsize, 1, top_k, 5]
        :param targets: list of tensors
        :param prediction: tensor [batchsize, top_k, num_classes+1]
        :return:
        """
        batchsize = rois.size(0)
        num_rois = rois.size(2)
        assign_labels = torch.zeros(size=(batchsize, num_rois)).long()
        # 给每一个roi按照与true的iou最大分配标签，如果iou小于阈值则让其为0背景
        if self.focal_loss:
            for idx in range(batchsize):
                assign_label_for_rois(rois[idx][0], targets[idx], assign_labels, idx, 0.3)
            loss = self.loss_func(prediction, assign_labels)

        else:
            for idx in range(batchsize):
                assign_label_for_rois(rois[idx][0], targets[idx], assign_labels, idx, 0.3)

            loss = 0
            for idx in range(batchsize):
                loss += self.loss_func(prediction[idx], assign_labels[idx])

        return loss / batchsize