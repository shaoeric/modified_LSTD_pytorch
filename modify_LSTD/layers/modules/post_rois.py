import torch
from torch.autograd import Function
from utils.box_utils import decode, nms
from config import voc as cfg
import cv2
import torch.nn as nn
import torchvision


class Post_rois(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.对loc解码，应用NMS，判断区域的objectness，不需要物体类别的信息。
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super(Post_rois, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        # Parameters used in nms.
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,source_num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, 2, self.top_k, 5)
        conf_data = self.softmax(conf_data)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance) # [8732，4]
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # [8732]
                scores = conf_scores[cl][c_mask]                # [n]
                # 如果当前图片当前类别的置信度都不大于阈值，则进行下一个类别的判断
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)  # [8732, 4]
                # boxes 为当前图片含有当前类别的置信度大于阈值的所有边界框 [n, 4]
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class, \
                # ids是针对scores的[n]而言的，而不是对[8732]的
                keep = torchvision.ops.nms(boxes, scores, self.nms_thresh)
                count = min(self.top_k, keep.size(0))
                output[i, cl, :count] = torch.cat((scores[keep[:count]].unsqueeze(1), boxes[keep[:count]]), 1)
        return output   # [batchsize, 2, N, 5]   5: [score， xmin, ymin, xmax, ymax]