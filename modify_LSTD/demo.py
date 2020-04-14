import torch
import torch.nn as nn
import cv2
from models.lstd_source_bd import build_ssd
from data import BaseTransform
from config import VOC_CLASSES as label, input_size, num_classes, top_k, cuda
from utils.box_utils import nms
import torchvision

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).cuda()
        x = torch.tensor(x.unsqueeze(0))
        scale = torch.Tensor([width, height, width, height]).cuda()
        with torch.no_grad():
            confidences, rois, objectness = net(x)  # forward pass
        rois = rois.cuda() if cuda else rois

        flag = True

        batch = confidences.size(0)
        output = torch.zeros(size=(batch, num_classes, top_k, 5)).to(confidences.device)
        for i in range(batch):
            roi = rois[i]
            confidence = confidences[i]
            for c in range(1, num_classes):
                score = confidence[:, c]
                if flag:
                    keep, count = nms(roi[0, :, 1:], score, top_k=top_k)
                    keep = keep[:count]
                else:
                    keep = torchvision.ops.nms(roi[0, :, 1:], score, 0.50)[:top_k]
                    count = keep.size(0)

                output[i, c, :count] = torch.cat((score[keep].unsqueeze(1), roi[0, keep, 1:]), 1)
                # print(torch.cat((score[keep].unsqueeze(1), roi[0, keep, 1:]), 1))
        if flag:
            for c in range(output.size(1)):
                r = 0
                while output[0, c, r, 0] > 1/(num_classes-1):
                    # print(output[0, c, r], c, r)
                    pt = (output[0, c, r, 1:] * scale).cpu().numpy()
                    cv2.rectangle(frame,
                                  (int(pt[0]), int(pt[1])),
                                  (int(pt[2]), int(pt[3])),
                                  COLORS[c % 3], 2)
                    cv2.putText(frame, label[c - 1], (int(pt[0]), int(pt[1])+50),
                                FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    r += 1
            return frame
        else:
            for i in range(output.size(0)):
                for r in range(output.size(2)):
                    max_score, idx = output[i, :, r, 0].max(-1)
                    if max_score.data > 1./ (num_classes-1):
                        pt = (output[i, idx, r, 1:] * scale).cpu().numpy()
                        cv2.rectangle(frame,
                                      (int(pt[0]), int(pt[1])),
                                      (int(pt[2]), int(pt[3])),
                                      COLORS[i % 3], 2)
                        cv2.putText(frame, label[idx - 1], (int(pt[0]), int(pt[1]+50)),
                                    FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return frame

    img = cv2.imread("E:\python_project\ssd\ssdpytorch\dataset\VOC\VOCdevkit\VOC2007\JPEGImages"
                     "\\000018.jpg")
    # img = cv2.imread("E:\\lena.jpg")
    frame = predict(img)
    cv2.imshow("", frame)
    cv2.waitKey(0)


weights = torch.load("./weights/trained/lstd_source1000.pth")
net = build_ssd('test', 300).cuda()
net = nn.DataParallel(net, device_ids=[0])
net.load_state_dict(weights)
transform = BaseTransform(input_size, (104/256.0, 117/256.0, 123/256.0))

cv2_demo(net.eval(), transform)
