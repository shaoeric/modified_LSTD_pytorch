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
            nn.Linear(config.conved_channel//8 * config.pooled_size **2, 64),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.25),
            nn.Linear(64, self.num_classes),
            nn.LeakyReLU(0.1, True)
        )

    # def forward(self, x):
        # x: [batch, num_roi, 16, 7, 7]
        # batchsize = x.size(0)
        # num_roi = x.size(1)
        # x = x.view(batchsize, num_roi, -1)
        # output = torch.zeros(size=(batchsize, num_roi, self.num_classes)).type(x.type())
        # for idx in range(batchsize):
        #     for j in range(num_roi):
        #         out = self.classifier(x[idx][j])
        #         output[idx, j, ...] = out
        # return output

    def forward(self, x, keep_count):
        # x: [batch, num_roi, 16, 7, 7]
        # keep_count: [batch]
        batchsize = x.size(0)
        num_roi = x.size(1)
        x = x.view(batchsize, num_roi, -1)
        output = torch.zeros(size=(batchsize, config.selected_proposal, self.num_classes)).type(x.type())
        for idx in range(batchsize):
            for j in range(min(keep_count[idx], config.selected_proposal)):
                out = self.classifier(x[idx][j])
                output[idx, j, ...] = out

        return output
