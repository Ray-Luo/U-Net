from torch import nn, optim
import torch
import os
import sys
import math
import numpy as np
import random
from dataset import UNetDataset
from operator import itemgetter 
from torchvision import transforms
from unet import UNet
from PIL import Image


def get_ids(data_folder):
    img_ids = list()
    gt_img_ids = list()
    for img in os.listdir(data_folder):
        if "gt" not in img:
            img_ids.append(data_folder+img)
        else:
            gt_img_ids.append(data_folder+img)
    img_ids.sort()
    gt_img_ids.sort()
    return img_ids, gt_img_ids

img_dim = 256
checkpoint_root = './checkpoint/'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

test_folder = './dataset/test/'
img_ids, gt_img_ids = get_ids(test_folder)

for i in range(len(img_ids)):
    print(img_ids[i], gt_img_ids[i])



test_data = UNetDataset(img_ids, gt_img_ids, img_dim)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)

unet = UNet()
unet = unet.to(device)

for file in os.listdir(checkpoint_root):
    if file.startswith("unet") and file.endswith(".tar"):
        checkpoint = torch.load(checkpoint_root + file, map_location='cpu')
        unet.load_state_dict(checkpoint['state_dict'])

unet.eval()
with torch.no_grad():
    for batch_idx, (img, gt_img) in enumerate(test_loader):
        img = img.to(device)
        gt_img = gt_img.type(torch.LongTensor).to(device)

        pred = unet(img)
        pred = torch.max(pred.data, 1)[1]
        print(pred.size())
        pred = pred.squeeze(0).cpu().data.numpy()
        print(pred.max(),pred.min())
        pred = (((pred - pred.min()) / (pred.max() - pred.min())) * 255.9).astype(np.uint8)
        image = Image.fromarray(pred)
        image.show()
        break

