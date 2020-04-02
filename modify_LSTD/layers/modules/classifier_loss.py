
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import assign_label_for_rois
import config

class ClassifierLoss(nn.Module):
    def __init__(self):
        super(ClassifierLoss, self).__init__()
        self.loss_func = F.cross_entropy

    def forward(self, rois, targets, prediction):
        """

        :param rois: tensor [batchsize, 1, top_k, 5]
        :param targets: list of tensors
        :param prediction: tensor [batchsize, top_k, num_classes+1]
        :return:
        """
        batchsize = rois.size(0)
        num_rois = rois.size(2)
        rois[:, :, :, 1:] = rois[:, :, :, 1:] / config.input_size * 16
        rois.clamp_(min=0., max=1.)
        assign_labels = torch.zeros(size=(batchsize, num_rois)).long()
        # 给每一个roi按照与true的iou最大分配标签，如果iou小于阈值则让其为0背景
        for idx in range(batchsize):
            assign_label_for_rois(rois[idx][0], targets[idx], assign_labels, idx, 0.5)

        loss = 0
        for idx in range(batchsize):
            loss += self.loss_func(prediction[idx], assign_labels[idx])

        return loss / batchsize