import torch
import torch.nn as nn
import cv2
from models.lstd_source_bd import build_ssd
from data import BaseTransform
from config import VOC_CLASSES as label, input_size, num_classes, top_k, cuda
from utils.box_utils import nms

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1).cuda()
        x = torch.tensor(x.unsqueeze(0))
        scale = torch.Tensor([width, height, width, height]).cuda()
        with torch.no_grad():
            confidence, rois = net(x)  # forward pass

        output = torch.zeros(size=(num_classes, top_k, 5))
        output = output.cuda() if cuda else output
        rois = rois.cuda() if cuda else rois
        for c in range(1, num_classes):
            id, count = nms(rois[0,0,:,1:].cpu(), confidence[0, :, c].cpu(), top_k=top_k)
            id = id.cuda() if cuda else id
            output[c, :count] = torch.cat((confidence[0, id[:count], c].unsqueeze(1), rois[0, 0, id[:count], 1:]), 1)

        for i in range(output.size(0)):
            j = 0
            while output[i, j, 0] >= 0.5:
                pt = (output[i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, label[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    img = cv2.imread("E:\\000000000785.jpg")
    frame = predict(img)
    cv2.imshow("", frame)
    cv2.waitKey(0)


weights = torch.load("./weights/trained/lstd_source114000.pth")
net = build_ssd('test', 300, 2).cuda()
net = nn.DataParallel(net, device_ids=[0])
# net.load_state_dict(weights)
transform = BaseTransform(input_size, (104/256.0, 117/256.0, 123/256.0))

cv2_demo(net.eval(), transform)
