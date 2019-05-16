import torch
import torch.onnx
import os
from unet import UNet

checkpoint_root = './checkpoint/'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

unet = UNet()
unet = unet.to(device)
dummy_input = torch.randn(1, 1, 256, 256, device=device)
for file in os.listdir(checkpoint_root):
    if file.startswith("unet") and file.endswith(".tar"):
        checkpoint = torch.load(checkpoint_root + file, map_location='cpu')
        unet.load_state_dict(checkpoint['state_dict'])


torch.onnx.export(unet, dummy_input, checkpoint_root + "onnx_unet.onnx")
