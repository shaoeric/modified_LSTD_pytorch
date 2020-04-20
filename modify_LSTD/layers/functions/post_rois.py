import torch
from torch.autograd import Function
from utils.box_utils import decode, nms
from config import voc as cfg
import cv2
import torch.nn as nn


class Post_rois(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.对loc解码，应用NMS，判断区域的objectness，不需要物体类别的信息。
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, loc_data, conf_data, prior_data):
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
        output = torch.zeros(num, 1, self.top_k, 5)
        conf_data = self.softmax(conf_data)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i][1].clone()
            c_mask = conf_scores.gt(self.conf_thresh)
            scores = conf_scores[c_mask]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            scores[:] = i
            output[i, 0, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        
        # Decode predictions into bboxes.
        # for i in range(num):
        #     assert prior_data.cpu().numpy().all() >= 0
        #     prior_data = prior_data.cuda(loc_data[i].get_device())
        #     decoded_boxes = decode(loc_data[i], prior_data, self.variance)
        #
        #     # For each class, perform nms
        #     conf_scores = conf_preds[i][1].clone()
        #     # filter
        #     # apply nms
        #     ids, count = nms(decoded_boxes, conf_scores, self.nms_thresh, 1000)
        #
        #     # sort all conf_scores from high to low
        #     sort_score, sort_index = torch.sort(conf_scores[ids[:count]], descending=True)
        #     # get top 100
        #     sort_index = sort_index[:self.top_k]
        #     scores = conf_scores[ids[:count]][sort_index]
        #     decoded_boxes = decoded_boxes[ids[:count]][sort_index, :]
        #     # change score to img index
        #     scores[:] = i
        #     # 只需要区分背景和物体，所以输出的不管物体类别的问题
        #     output[i, 0, ...] = torch.cat((scores.unsqueeze(1), decoded_boxes), 1)


        return output  # [batchsize, 1, N, 5]   5: [图像id， xmin, ymin, xmax, ymax]