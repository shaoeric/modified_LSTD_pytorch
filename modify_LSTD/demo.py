import torch
import torch.nn as nn
import cv2
from models.lstd_source import build_ssd
from data import BaseTransform
from config import VOC_CLASSES as label, input_size, num_classes, selected_proposal, cuda
from utils.box_utils import nms
import torchvision
import config

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).cuda()
        x = torch.tensor(x.unsqueeze(0))
        scale = torch.Tensor([width, height, width, height]).cuda()
        with torch.no_grad():
            confidences, rois = net(x, True)  # forward pass
        rois = rois.cuda() if cuda else rois

        batch = confidences.size(0)
        output = torch.zeros(size=(batch, num_classes, selected_proposal, 5)).to(confidences.device)
        for i in range(batch):
            roi = rois[i]
            confidence = confidences[i]
            for c in range(1, num_classes):
                score = confidence[:, c]
                keep = torchvision.ops.nms(roi[0, :, :], score, 0.45)[:selected_proposal]
                count = keep.size(0)
                output[i, c, :count] = torch.cat((score[keep].unsqueeze(1), roi[0, keep, :]), 1)

        for i in range(output.size(0)):
            for r in range(output.size(2)):
                print(output[i, :, r, :])
                max_score, idx = output[i, :, r, 0].max(-1)
                if max_score.data >= 0.18:
                    pt = (output[i, idx, r, 1:] * scale).cpu().numpy()
                    cv2.rectangle(frame,
                                  (int(pt[0]), int(pt[1])),
                                  (int(pt[2]), int(pt[3])),
                                  COLORS[r % 3], 2)
                    cv2.putText(frame, label[idx - 1], (int(pt[0]), int(pt[1])), FONT, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    img = cv2.imread("E:\python_project\ssd\ssdpytorch\dataset\VOC\VOCdevkit\VOC2007\JPEGImages"
                     "\\000030.jpg")
    # img = cv2.imread("E:\\lena.jpg")
    frame = predict(img)
    cv2.imshow("", frame)
    cv2.waitKey(0)


weights = torch.load("./weights/trained/lstd_source12500.pth", map_location='cuda:0')
net = build_ssd('test', 300).cuda()
net.load_state_dict(weights)
transform = BaseTransform(input_size, (104/256.0, 117/256.0, 123/256.0))

cv2_demo(net.eval(), transform)
