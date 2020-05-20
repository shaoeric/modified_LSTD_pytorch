# Few-Shot Object Detection based on LSTD framework

This work is my undergraduation graduation project, which mainly studies few-shot object detection and then reproduces and modifies the  framework based on LSTD.

## Getting Started

- source-domain dataset: VOC PASCAL 07&12
- target-domain dataset: customized dataset, 15 samples with full annotations for each category are almost enough 

### Installing

- torch 1.4.0
- torchvision 0.5.0
- opencv-python 4.1.2.30
- Pillow 7.0.0
- cuda 10.1

### Config

1. prepare your target-domain dataset
2. specified your configuration in config.py, including target-domain path, target_num_classes and target_classes

## Train source-domain model

LSTD requires transfering knowledge from source-domain to target-domain, it is necessary to train on source-domain dataset.

```
python train.py
```

where batch_size=16 is recommanded

## Train target-domain model

```
python train_target.py
```

## Demo

1. specify your image path

2. specify your weight path

3. ```
   python demo.py
   ```
## pretrained weights

[download from baidunetdisk](https://pan.baidu.com/s/1tzHk0g_M42KH9dGNHF8XCg)    code：op7i

## Modification

- generative mask background suppression

  It reduces the dimension  of  thick feature cube with statistical methods to obtain a thin feature map of the mininum, maximum, average and variance matrice stack. And then use convolutional self-encoder network to generate the mask as its background suppression regularization.

- hot start classification training mechanism

  First, finetune the RPN network on the target-domain dataset, and freeze the ROI layers and cls layers.  When the training process meets certain conditions, start to train the whole framework.

## Result

On the customized dataset, mAP of the modified LSTD is 0.4 higher than that of origin LSTD.

Some results of test images are available in the result filefold.

Beside, a LOL video game is tested with default parameters, [bilibili video link](https://www.bilibili.com/video/BV1Gz4y1d7r3?from=search&seid=3337204679427424785)

## Acknowledgments

* (ssd.pytorch)[https://github.com/amdegroot/ssd.pytorch]
* (LSTD)[https://arxiv.org/abs/1803.01529]

