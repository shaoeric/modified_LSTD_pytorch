import torch
import torch.nn as nn
import cv2
from models.lstd_target import build_ssd as build_target
from models.lstd_source import build_ssd as build_source
from data import BaseTransform
from config import VOC_CLASSES as voc_label, TARGET_CLASSES as label, input_size, source_num_classes, selected_proposal, cuda
from utils.box_utils import nms
import torchvision
import config

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def predict(net, frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).cuda()
    x = torch.tensor(x.unsqueeze(0))
    scale = torch.Tensor([width, height, width, height]).cuda()
    with torch.no_grad():
        detections, mask = net(x, True)  # forward pass

    for i in range(detections.size(0)):
        for r in range(detections.size(2)):
            max_score, idx = detections[i, :, r, 0].max(-1)
            if max_score.data >= 0.4:
                pt = (detections[i, idx, r, 1:] * scale).cpu().numpy()
                if pt[2] <= pt[0] or pt[3] <= pt[1]: continue
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[r % 3], max(2 * height // 480, 2))
                cv2.putText(frame, label[idx - 1], (int(pt[0]), int(pt[1]+20)), FONT, 1.5 * height / 480, (0, 255, 255), max(2 * height // 480, 2), cv2.LINE_AA)
    return frame, mask



weights = torch.load("./weights/trained/game1800.pth", map_location='cuda:0')
net = build_target('detect', 300).cuda()
net.load_state_dict(weights)
transform = BaseTransform(input_size, (104/256.0, 117/256.0, 123/256.0))
net.eval()

import os
root = 'E:\python_project\ssd\\video\\temp_pic'
result_root = 'E:\\python_project\\ssd\\video\\result'
for filename in os.listdir(root):
    img = os.path.join(root, filename)
    img = cv2.imread(img)
    result, _ = predict(net, img)
    cv2.imwrite(os.path.join(result_root, filename), result)
    # cv2.imwrite('result/'+filename, mask[0][0].cpu().numpy() * 255)
