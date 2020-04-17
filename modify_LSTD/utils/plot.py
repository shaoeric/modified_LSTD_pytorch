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
    tensor = tensor[batch_index][channel_index]
    img = tensor.clone()
    with torch.no_grad():
        img = img.cpu().numpy()
        plt.imshow(img)
        plt.show()


def show_tensor(tensor:torch.Tensor, batch_index):
    tensor = tensor[batch_index]
    img = tensor.clone()
    with torch.no_grad():
        img = img.cpu().numpy()
        img = np.transpose(img, [1,2,0])
        plt.imshow(img)
        plt.show()


def show_tensor_average(tensor:torch.Tensor, bath_index):
    img = tensor[bath_index]
    img = torch.mean(img, dim=0)
    img = img.cpu().numpy()
    plt.imshow(img)
    plt.show()