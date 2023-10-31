from __future__ import annotations

import os 
from os import PathLike
from pathlib import Path
from typing import Callable, Tuple, Union

from PIL import Image

import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from sklearn.model_selection import train_test_split

def split_data(dataset:Dataset, test_size:float=0.2) -> Tuple[Subset, Subset]:
    """Split a dataset into training and test subsets."""
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=test_size
    )
    dataset_train = Subset(dataset, indices=train_idx)
    dataset_test = Subset(dataset, indices=test_idx)
    return dataset_train, dataset_test

def subset_debug(dataset:Dataset, n:int=10) -> Subset:
    """Take a small subset of data for debugging purposes."""
    idxs = list(range(len(dataset)))
    debug_dataset = Subset(dataset, indices=idxs[:n])
    return debug_dataset

def get_transform(train) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply image transormations, if train perform image dynamic augmentation."""
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class CustomDataset(Dataset): 
    """Custom dataset class for the Penn-Fudan image segmentation dataset."""
    def __init__(self, root:Union[str, PathLike], transforms=None):
        """Penn-Fudan Dataset. 

        Parameters
        ----------
        root : Union[str, PathLike]
            Path to the root directory containing the dataset files/folders.
        transforms : callable, optional
            Image transformations (e.g., torchvision.transforms.Compose), by default 
            None
        
        Notes
        -----
        Dataset contains 170 images with 345 intances. 
        
        image: torchvision.tv_tensors.Image of shape [3, H, W], a pure tensor, or a 
        PIL Image of size (H, W)

        target: a dict containing the following fields

            boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]:
                the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
                ranging from 0 to W and 0 to H

            labels, integer torch.Tensor of shape [N]: the label for each bounding box. 
                0 represents always the background class.

            image_id, int: an image identifier. It should be unique between all the 
                images in the dataset, and is used during evaluation

            area, float torch.Tensor of shape [N]: the area of the bounding box. This 
                is used during evaluation with the COCO metric, to separate the metric 
                scores between small, medium and large boxes.

            iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will 
                be ignored during evaluation.

            (optionally) masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the 
                segmentation masks for each one of the objects
        """
        self.root = str(root)

        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)

        # instances are encoded as different colors
        # NOTE: [C, H, W], where C ∈ Z:{0, 1, 2, ..., N} repr each class.   
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        # NOTE: [C=1, H, W] -> [C=N, H, W], where N is the number of instances
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class (i.e., a pedestrian)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = dict()
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
