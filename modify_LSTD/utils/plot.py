import matplotlib.pyplot as plt
import torch
import numpy as np

def show_tensor_gray(tensor: torch.Tensor, batch_index: int, channel_index:int):
    """
    :param tensor: size [batchsize, channels, h, w]
    :param batch_index:
    :param channel_index
    :return:
    """
    img = tensor[batch_index][channel_index]
    img = img.numpy()
    plt.imshow(img)
    plt.show()

def show_tensor(tensor:torch.Tensor, batch_index):
    img = tensor[batch_index]
    img = img.numpy()
    img = np.transpose(img, [1,2,0])
    plt.imshow(img)
    plt.show()

