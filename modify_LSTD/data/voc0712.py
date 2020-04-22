import config
import sys
import torch
from torch.utils.data import Dataset
import cv2
import os
import os.path as osp
import numpy as np
if sys.version_info[0] == 2: # python version
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None, keep_difficult=False):
        if class_to_ind is not None:
            if isinstance(class_to_ind, tuple):
                self.class_to_ind = dict(zip(class_to_ind, range(len(class_to_ind))))
            elif isinstance(class_to_ind, dict):
                self.class_to_ind = class_to_ind
        else:
            self.class_to_ind = class_to_ind or dict(
                zip(config.VOC_CLASSES, range(len(config.VOC_CLASSES))))

        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
            scaled: return a list of scaled coordinates or not
        Returns:
            a list containing lists of bounding boxes
            [[xmin, ymin, xmax, ymax, label_ind], ... ]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712', mask=True):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.mask = mask
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, img_mask = self.pull_item(index)
        return im, gt, img_mask

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        img_mask = None
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  # 相对于原图的变换，而不是相对于300x300
        if self.transform is not None:
            target = np.array(target)
            # 绘制蒙版
            if self.mask:
                # 当前img为原尺寸，target是相对于原尺寸的scale，要得到mask
                img_mask = self.get_mask(height, width, target)

            # 数据增强变换，让蒙版也参与变换
            img, boxes, labels, img_mask = self.transform(img, target[:, :4], target[:, 4], img_mask)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # 如果mask为True，返回蒙版 蒙版是对应的38x38特征图的蒙版，否则返回None
            img_mask = cv2.resize(img_mask, (config.voc['feature_maps'][0], config.voc['feature_maps'][0]))
            img_mask = torch.from_numpy(img_mask).unsqueeze(0) if self.mask else None
            img = torch.from_numpy(img).permute(2, 0, 1)
        return img, target, height, width, img_mask

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def get_mask(self, h, w, targets):
        """
        绘制图像的蒙版
        :param h: 原图像的高
        :param w: 原图像的宽
        :param targets: 归一化后的坐标以及label
        :return: 原图像的蒙版，背景区域为黑色，物体区域为白色
        """
        mask = np.zeros(shape=(h, w))
        for target in targets:
            x_min, y_min, x_max, y_max, _ = target  # scaled
            x_min *= w
            x_max *= w
            y_min *= h
            y_max *= h
            mask[int(y_min): int(y_max), int(x_min): int(x_max)] = 1  # 数据训练的时候使用1，但是在显示出来的时候需要把1变成255
        return mask.astype(np.uint8)


class CustomDataset(Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root, annotation_path,
                 transform=None, target_transform=VOCAnnotationTransform(class_to_ind=config.TARGET_CLASSES), mask=True):
        self.root = root
        self.annotation_path = annotation_path
        self.transform = transform
        self.target_transform = target_transform
        self.mask = mask
        self.ids = list()
        for file in os.listdir(root):
            filename, _ = osp.splitext(file)
            self.ids.append(filename)

    def __getitem__(self, index):
        im, gt, h, w, img_mask = self.pull_item(index)
        return im, gt, img_mask

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(osp.join(self.annotation_path, img_id+".xml")).getroot()
        img = cv2.imread(osp.join(self.root, img_id+".jpg"))
        height, width, channels = img.shape

        img_mask = None
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  # 相对于原图的变换，而不是相对于300x300
        if self.transform is not None:
            target = np.array(target)
            # 绘制蒙版
            if self.mask:
                # 当前img为原尺寸，target是相对于原尺寸的scale，要得到mask
                img_mask = self.get_mask(height, width, target)
            # 数据增强变换，让蒙版也参与变换
            img, boxes, labels, img_mask = self.transform(img, target[:, :4], target[:, 4], img_mask)

            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # 如果mask为True，返回蒙版 蒙版是对应的38x38特征图的蒙版，否则返回None
            img_mask = cv2.resize(img_mask, (config.voc['feature_maps'][0], config.voc['feature_maps'][0]))

            img_mask = torch.from_numpy(img_mask).unsqueeze(0) if self.mask else None
            img = torch.from_numpy(img).permute(2, 0, 1)

        return img, target, height, width, img_mask


    def get_mask(self, h, w, targets):
        """
        绘制图像的蒙版
        :param h: 原图像的高
        :param w: 原图像的宽
        :param targets: 归一化后的坐标以及label
        :return: 原图像的蒙版，背景区域为黑色，物体区域为白色
        """
        mask = np.zeros(shape=(h, w))
        for target in targets:
            x_min, y_min, x_max, y_max, _ = target  # scaled
            x_min *= w
            x_max *= w
            y_min *= h
            y_max *= h
            mask[int(y_min): int(y_max), int(x_min): int(x_max)] = 1  # 数据训练的时候使用1，但是在显示出来的时候需要把1变成255
        return mask.astype(np.uint8)


if __name__ == '__main__':
    from utils.auguments import SSDAugmentation
    from torch.utils.data import DataLoader
    from data import detection_collate
    # dataset = VOCDetection(config.VOC_ROOT, transform=SSDAugmentation(config.voc['min_dim']),mask=True)
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=detection_collate,pin_memory=True)
    # iterator = iter(dataloader)
    # # images:[batchsize, channels, h, w]
    # # targets:list of tensor[number_bbox, 5]
    # # masks:[batchsize, 1, h, w]
    # images, targets, masks = next(iterator)
    # print(images.requires_grad)  # False
    # # print(targets)

    dataset = CustomDataset(config.target_VOC_ROOT, config.target_VOC_Annotations, transform=SSDAugmentation(config.voc['min_dim']), mask=True)
    dataloader = DataLoader(dataset, batch_size=2,pin_memory=True, collate_fn=detection_collate)
    iterator = iter(dataloader)
    while True:
        try:
            images, targets, masks = next(iterator)
        except StopIteration as e:
            print("*"*30)
            iterator = iter(dataloader)