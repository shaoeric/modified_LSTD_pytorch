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

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh):
        super(Post_rois, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  # 8732
        output = torch.zeros(num, 1, self.top_k, 5)
        conf_data = self.softmax(conf_data)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i][1].clone()
            c_mask = conf_scores.gt(self.conf_thresh)
            scores = conf_scores[c_mask]

            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4).clamp_(0, 1)
            keep = torchvision.ops.nms(boxes, scores, 0.4)
            scores[:] = i
            output[i, 0, :min(self.top_k, keep.size(0))] = torch.cat((scores[keep[:self.top_k]].unsqueeze(1), boxes[keep[:self.top_k]]), 1)

        return output  # [batchsize, 1, N, 5]   5: [图像id， xmin, ymin, xmax, ymax]