from torch.autograd import Function


class MaskBlock(Function):
    def __init__(self):
        super(MaskBlock, self).__init__()
        pass

    def forward(self, img, mask):
        """
        使用mask对img进行掩盖，物体区域显示物体，背景区域为黑色
        :param img: [batchsize, channels, h, w]
        :param mask:[batchsize, 1, h, w]
        :return:
        """
        idx = mask.expand_as(img).eq(0)
        img[idx] = 0
        return img
