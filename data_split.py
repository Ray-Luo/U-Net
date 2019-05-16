import os
import random

data_path = "./dataset/all_data/"


def get_ids(data_folder):
    img_ids = list()
    gt_img_ids = list()
    for img in os.listdir(data_folder):
        if "gt" not in img:
            img_ids.append(data_folder+img)
        else:
            gt_img_ids.append(data_folder+img)
    return img_ids, gt_img_ids

img_ids, gt_img_ids = get_ids(data_path)

indices = [i for i in range(len(img_ids))]

test_size = int(0.2 * len(indices))
test_set = set()
random.seed(123)
while (len(test_set) < test_size):
    test_set.add(random.choice(indices))

# print(len(test_set), len(img_ids))
# print(len(test_set))
training_ids = set(indices) - test_set
# print(len(training_ids), len(img_ids))



data_folder = "./dataset/all_data/"
img_ids, gt_img_ids = get_ids(data_folder)
img_ids.sort()
gt_img_ids.sort()

# for i in range(len(img_ids)):
#     print(img_ids[i], gt_img_ids[i])

ids = [i for i in range(len(img_ids))]

test_size = int(0.1 * len(ids))
test_ids = set()
random.seed(123)
while (len(test_ids) < test_size):
    test_ids.add(random.choice(ids))

# print(len(test_set), len(img_ids))
# print(len(test_set))
training_ids = list(set(ids) - test_ids)
test_ids = list(test_ids)

training_imgs = itemgetter(*training_ids)(img_ids)
training_gts = itemgetter(*training_ids)(gt_img_ids)

test_imgs = itemgetter(*test_ids)(img_ids)
test_gts = itemgetter(*test_ids)(gt_img_ids)

test_folder = "/home/kdd/Documents/U-Net/dataset/test/"
train_folder = "/home/kdd/Documents/U-Net/dataset/train/"

for file in test_imgs:
    name = file.split("/")[-1]
    copyfile(file, test_folder + name)
for file in test_gts:
    name = file.split("/")[-1]
    copyfile(file, test_folder + name)


for file in training_imgs:
    name = file.split("/")[-1]
    copyfile(file, train_folder + name)
for file in training_gts:
    name = file.split("/")[-1]
    copyfile(file, train_folder + name)

print("dataset finished")