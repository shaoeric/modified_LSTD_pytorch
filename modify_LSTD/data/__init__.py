import torch


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
            3) (tensor)
    """
    targets = []
    imgs = []
    masks = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        if sample[2] is not None:
            masks.append(sample[2])
    if len(masks):
        return torch.stack(imgs, 0), targets, torch.stack(masks, 0)
    else:
        return torch.stack(imgs, 0), targets, None