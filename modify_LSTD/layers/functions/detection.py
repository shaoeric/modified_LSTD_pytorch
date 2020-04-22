import torch
from torch.autograd import Function
from config import voc as cfg
from utils.box_utils import decode, nms
import torch.nn.functional as F
import torchvision


class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')

    def forward(self, confidences, rois):
        """

        :param confidences:[batch, num_rois, num_classes]
        :param rois: [batch, 1, num_rois, 4]
        :return: output :[batch, num_classes, num_rois, 5]
        """
        batch = confidences.size(0)
        output = torch.zeros(size=(batch, self.num_classes, self.top_k, 5)).to(confidences.device)
        for i in range(batch):
            roi = rois[i]
            confidence = confidences[i]
            for c in range(1, self.num_classes):
                score = confidence[:, c]
                keep = torchvision.ops.nms(roi[0, :, :], score, self.nms_thresh)[:self.top_k]
                count = keep.size(0)
                output[i, c, :count] = torch.cat((score[keep].unsqueeze(1), roi[0, keep, :]), 1)
        return output
