from torch.autograd import Function


class MaskBlock(Function):
    def __init__(self, is_bd=False):
        super(MaskBlock, self).__init__()
        self.is_bd = is_bd

    def forward(self, img, mask):
        """
        如果不是背景抑制，使用mask对img进行掩盖，物体区域显示物体，背景区域为黑色
        如果是背景抑制，使用mask对img进行掩盖，物体区域为黑色，背景区域不变
        :param img: [batchsize, channels, h, w]
        :param mask:[batchsize, 1, h, w]
        :return:
        """
        if not self.is_bd:
            idx = (mask.expand_as(img)).eq(0)
        else:
            idx = (mask.expand_as(img)).eq(1)

        img[idx] = 0
        return img
