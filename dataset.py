import torch
from torch.utils import data
import PIL
from PIL import Image
from PIL import ImageOps
import numpy as np
from torchvision import datasets, transforms

class UNetDataset(data.Dataset):

    def __init__(self, ids, labels, expected_img_size):
        self.ids = ids
        self.labels = labels
        self.expected_img_size = expected_img_size
        self.data_transforms = transforms.Compose([
            transforms.Resize((expected_img_size,expected_img_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # # transforms.RandomPerspective(),
            # transforms.RandomRotation(degrees=45, resample=PIL.Image.BILINEAR),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        img = Image.open(img_path)
        img = self.data_transforms(img)
        img = np.array(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        # print(img.max(),img.min())

        gt_img_path = self.labels[index]
        gt_img = Image.open(gt_img_path)
        gt_img = self.data_transforms(gt_img)
        gt_img = np.array(gt_img)
        # print(gt_img.max(),gt_img.min())
        gt_img = torch.from_numpy(gt_img)

        return (img, gt_img)