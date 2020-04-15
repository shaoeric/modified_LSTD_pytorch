# config.py
import os.path
import os
# gets home dir cross platform
HOME = os.path.dirname(__file__)
import torch

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

VOC_ROOT = "E:/python_project/ssd/ssdpytorch/dataset/VOC/VOCdevkit"
basenet = 'vgg16_reducedfc.pth'

batch_size = 1
num_workers = 4
lr = 2e-4
momentum = 0.9
weight_decay = 1e-4
gamma = 0.1
pretrained_folder = os.path.join(HOME, 'weights', 'pretrained')
save_folder = os.path.join(HOME, 'weights', 'trained')
cuda = True if torch.cuda.is_available() else False

num_classes = 21
mask_thresh = 0.3
top_k = 1000
selected_proposal = 500
conf_thresh = 0.01
rpn_nms_thresh = 0.7
nms_thresh = 0.45
pooled_size = 7
conved_channel = 128
input_size = 300

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (3000, 6000, 8000),
    'max_iter': 12000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
