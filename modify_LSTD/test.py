import torch
import torch.nn as nn
from models.lstd import build_ssd


net = build_ssd("test").cuda()
# net.eval()
img = torch.rand(size=(2, 3, 300, 300)).cuda()
net(img)

# net.print_params_num()