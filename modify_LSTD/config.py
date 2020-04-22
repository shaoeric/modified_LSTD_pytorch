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
target_VOC_ROOT = "E:\python_project\ssd\ssdpytorch\dataset\\custom_dataset\\train"
target_VOC_Annotations = "E:\python_project\ssd\ssdpytorch\dataset\\custom_dataset\\annotations\\train"

basenet = 'vgg16_reducedfc.pth'

pretrained_folder = os.path.join(HOME, 'weights', 'pretrained')
save_folder = os.path.join(HOME, 'weights', 'trained')
cuda = True if torch.cuda.is_available() else False
device = 'cuda:0'

batch_size = 2
num_workers = 4
lr = 5e-4
momentum = 0.9
weight_decay = 1e-4
gamma = 0.1

source_num_classes = 20+1
target_num_classes = 5+1
mask_thresh = 0.3
top_k = 200
selected_proposal = 100
conf_thresh = 0.01
rpn_nms_thresh = 0.9
rpn_iou_label_thresh = 0.3
rpn_train_max_iteration = 4000
classifier_iou_label_thresh = 0.3
nms_thresh = 0.45
pooled_size = 7
conved_channel = 128
input_size = 300


# SSD300 CONFIGS
voc = {
    'source_num_classes': 21,
    'lr_steps': (5000, 8000, 10000),
    'max_iter': 13010,
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

print(
    "config:\nbatchsize = {}\nlr = {}\ntop_k = {}\nselected_propsal = {}\nconf_thresh = {}\nrpn_nms_thresh = {}\nrpn_iou_label_thresh = {}\nrpn_train_max_iteration = {}\nclassifier_iou_label_thresh = {}\nnms_thresh = {}\npooled_size = {}\nconved_channel = {}\nlr_steps = {}".format(
        batch_size, lr, top_k, selected_proposal, conf_thresh,
        rpn_nms_thresh, rpn_iou_label_thresh, rpn_train_max_iteration,
        classifier_iou_label_thresh, nms_thresh, pooled_size, conved_channel, voc['lr_steps']
    )
)

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

TARGET_CLASSES = (
    'bear', 'elephant', 'kite', 'laptop', 'truck'
)

coco = {
    'source_num_classes': 201,
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
