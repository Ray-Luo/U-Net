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
from sklearn.model_selection import KFold
from unet import UNet
from shutil import copyfile


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
iters = 200
log_interval = 5
checkpoint_root = './checkpoint/'
test_folder = './dataset/test/'
train_folder = './dataset/train/'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

test_imgs, test_gts = get_ids(test_folder)
train_imgs, train_gts = get_ids(train_folder)


# for i in range(len(train_imgs)):
#     print(train_imgs[i], train_gts[i])

# for i in range(len(test_imgs)):
#     print(test_imgs[i], test_gts[i])



test_data = UNetDataset(test_imgs, test_gts, img_dim)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True, num_workers=2)


all_train_losses = list()
all_vali_losses = list()
all_test_losses = list()
min_loss = float("inf")


# load last checkpoint
# for file in os.listdir(checkpoint_root):
#     if file.startswith("unet") and file.endswith(".tar"):
#         checkpoint = torch.load(checkpoint_root + file, map_location='cpu')
#         unet.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         all_train_losses = checkpoint['train_losses']
#         all_test_losses = checkpoint['test_losses']
#         min_loss = checkpoint['min_loss']
#         epoch = checkpoint['epoch']


kf = KFold(n_splits=5, shuffle=True, random_state=123)
fold = -1

for train_index, vali_index in kf.split(train_imgs):
    unet = UNet()
    unet = unet.to(device)
    # print(unet)
    optimizer = optim.Adam(unet.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    train_losses = list()
    vali_losses = list()
    test_losses = list()

    fold += 1
    # print(train_index, vali_index)

    fd_training_imgs, fd_vali_imgs = itemgetter(*train_index)(train_imgs), itemgetter(*vali_index)(train_imgs)
    fd_training_gts, fd_vali_gts = itemgetter(*train_index)(train_gts), itemgetter(*vali_index)(train_gts)

    train_data = UNetDataset(fd_training_imgs, fd_training_gts, img_dim)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True, num_workers=2)

    vali_data = UNetDataset(fd_vali_imgs, fd_vali_gts, img_dim)
    vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=20, shuffle=True, num_workers=2)

    epoch = 0
    while epoch < iters:
        unet.train()
        epoch_train_loss = list()

        for batch_idx, (img, gt_img) in enumerate(train_loader):
            img = img.to(device)
            gt_img = gt_img.type(torch.LongTensor).to(device)

            pred = unet(img)
            loss = criterion(pred, gt_img)
            epoch_train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Fold: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.4f}'.format(
                    fold, epoch, batch_idx * len(img), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item()))

        mean_train_loss = np.mean(np.array(epoch_train_loss))
        train_losses.append(mean_train_loss)
        print('=================================================> Fold: {} Epoch: {} Average training_loss: {:.4f}'.format(fold, epoch, mean_train_loss))


        unet.eval()
        epoch_vali_loss = list()
        with torch.no_grad():
            for batch_idx, (img, gt_img) in enumerate(vali_loader):
                img = img.to(device)
                gt_img = gt_img.type(torch.LongTensor).to(device)

                pred = unet(img)
                loss = criterion(pred, gt_img)
                epoch_vali_loss.append(loss.item())

            mean_vali_loss = np.mean(np.array(epoch_vali_loss))
            vali_losses.append(mean_vali_loss)
            print('=================================================> Fold: {} Epoch: {} Average   valida_loss: {:.4f}'.format(fold, epoch, mean_vali_loss))

            
            epoch_test_loss = list()
            for batch_idx, (img, gt_img) in enumerate(test_loader):
                img = img.to(device)
                gt_img = gt_img.type(torch.LongTensor).to(device)

                pred = unet(img)
                loss = criterion(pred, gt_img)
                epoch_test_loss.append(loss.item())

            mean_test_loss = np.mean(np.array(epoch_test_loss))
            test_losses.append(mean_test_loss)
            print('=================================================> Fold: {} Epoch: {} Average  testing_loss: {:.4f}'.format(fold, epoch, mean_test_loss))


        if mean_vali_loss < min_loss:
            min_loss = mean_vali_loss

            for file in os.listdir(checkpoint_root):
                if file.startswith("unet") and file.endswith(".tar"):
                    os.remove(checkpoint_root + file)

            torch.save({
                    'epoch':         epoch,
                    'state_dict':    unet.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'train_losses':  train_losses,
                    'test_losses':   test_losses,
                    'min_loss':      min_loss
                    }, checkpoint_root + "unet_" + '{:.4f}'.format(min_loss) + ".tar")

        epoch += 1

    all_train_losses.append(train_losses)
    all_vali_losses.append(vali_losses)
    all_test_losses.append(test_losses)


torch.save({
    'train_losses':  all_train_losses,
    'vali_losses':   all_vali_losses,
    'test_losses':   all_test_losses,
    }, checkpoint_root + "losses.tar")




