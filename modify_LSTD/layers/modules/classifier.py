import torch
import torch.nn as nn
import config

class Classifier(nn.Module):
    def __init__(self, num_classes=21):
        """

        :param num_classes: 最后输出的个数 K+1
        """
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(config.conved_channel * config.pooled_size **2, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        # x: [batch, num_roi, 128, 7, 7]
        batchsize = x.size(0)
        num_roi = x.size(1)
        x = x.view(batchsize, num_roi, -1)
        output = torch.zeros(size=(batchsize, num_roi, self.num_classes)).type(x.type())
        for idx in range(batchsize):
            for j in range(num_roi):
                out = self.classifier(x[idx][j])
                output[idx, j, ...] = out
        return output
