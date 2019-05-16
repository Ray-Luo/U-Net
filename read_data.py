import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import png

save_path = "/home/ray/Documents/U-Net/dataset/test/"

        # print (os.path.join(subdir, file))
def process(file, path):
    img = nib.load(path).get_fdata()
    print(img.shape)
    for i in range(img.shape[2]):
        arr = img[:,:,i]
        arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255.9).astype(np.uint8)
        # arr = arr.astype(np.uint8)
        # image = Image.fromarray(arr)
        # im.show()
        # print(img[:,:,i])
        # png.from_array(np.uint8(img[:,:,i] * 255), 'L').save(save_path + "patient002_" + str(i) + ".png")
        # plt.plot(img[:,:,i])
        # fig = plt.imshow(img[:,:,i], cmap="gray")
        # plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        image = Image.fromarray(arr,'L')
        image.save(save_path + file[:-4] + "_" + str(i) + ".png")
        # plt.savefig(save_path + "patient002_" + str(i) + ".png", bbox_inches='tight', pad_inches = 0)

def processGT(file, path):
    img = nib.load(path).get_fdata()
    print(img.shape)
    for i in range(img.shape[2]):
        arr = img[:,:,i]
        # arr = (((arr - arr.min()) / (arr.max() - arr.min())) * 255.9).astype(np.uint8)
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr)
        # im.show()
        # print(img[:,:,i])
        # png.from_array(np.uint8(img[:,:,i] * 255), 'L').save(save_path + "patient002_" + str(i) + ".png")
        # plt.plot(img[:,:,i])
        # fig = plt.imshow(img[:,:,i], cmap="gray")
        # plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # image = Image.fromarray(np.uint8(img[:,:,i] * 255),'L')
        image.save(save_path + file[:-4] + "_" + str(i) + ".png")

for subdir, dirs, files in os.walk("/home/ray/Documents/U-Net/dataset/train/"):
    for file in files:
        print(file)
        path = os.path.join(subdir, file)
        if file.endswith("nii") and "_4d" not in file:
            if "_gt" not in file:
                process(file, path)
            else:
                processGT(file, path)


# img = nib.load("/home/ray/Documents/U-Net/dataset/train/patient004/patient004_frame15_gt.nii").get_fdata()
# for i in range(10):
#     # plt.plot(img[:,:,i])
#     # fig = plt.imshow(img[:,:,i], cmap="gray")
#     # plt.axis('off')
#     # fig.axes.get_xaxis().set_visible(False)
#     # fig.axes.get_yaxis().set_visible(False)
#     # # image = Image.fromarray(np.uint8(img[:,:,i] * 255),'L')
#     # plt.savefig(save_path + "patient002_gt_" + str(i) + ".png", bbox_inches='tight', pad_inches = 0)
#     arr = img[:,:,i]
#     assert arr.any() > 0 
#     arr = arr.astype(np.uint8)
#     print(arr.max(),arr.min())
#     image = Image.fromarray(arr,'L')
#     image.save(save_path + "patient004_frame15_gt_" + str(i) + ".png")