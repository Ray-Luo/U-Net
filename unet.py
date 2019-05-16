import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=1, \
                               out_channels=64, \
                               kernel_size=3, \
                               stride=1, padding=1, bias=True)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv21 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=True)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv31 = nn.Conv2d(128, 256, 3, 1, padding=1, bias=True)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv41 = nn.Conv2d(256, 512, 3, 1, padding=1, bias=True)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv51 = nn.Conv2d(512, 1024, 3, 1, padding=1, bias=True)
        self.conv52 = nn.Conv2d(1024, 1024, 3, 1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(1024)
        self.transConv5 = nn.ConvTranspose2d(in_channels=1024, \
                                            out_channels=512, \
                                            kernel_size=2, \
                                            stride=2, \
                                            bias=True)

        self.conv61 = nn.Conv2d(1024, 512, 3, 1, padding=1, bias=True)
        self.conv62 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(512)
        self.transConv6 = nn.ConvTranspose2d(512, 256, 2, 2,  bias=True)

        self.conv71 = nn.Conv2d(512, 256, 3, 1, padding=1, bias=True)
        self.conv72 = nn.Conv2d(256, 256, 3, 1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(256)
        self.transConv7 = nn.ConvTranspose2d(256, 128, 2, 2, bias=True)

        self.conv81 = nn.Conv2d(256, 128, 3, 1, padding=1, bias=True)
        self.conv82 = nn.Conv2d(128, 128, 3, 1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(128)
        self.transConv8 = nn.ConvTranspose2d(128, 64, 2, 2, bias=True)

        self.conv91 = nn.Conv2d(128, 64, 3, 1, padding=1, bias=True)
        self.conv92 = nn.Conv2d(64, 64, 3, 1, padding=1, bias=True)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv93 = nn.Conv2d(64, 4, 1, 1)
        self.bn10 = nn.BatchNorm2d(4)

        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        print('x', x.size())
        block11 = self.relu(self.bn1(self.conv12(self.relu(self.bn1(self.conv11(x))))))
        print('block11', block11.size())
        block12 = self.maxpool1(block11)
        print('block12', block12.size())

        block21 = self.relu(self.bn2(self.conv22(self.relu(self.bn2(self.conv21(block12))))))
        print('block21', block21.size())
        block22 = self.maxpool2(block21)
        print('block22', block22.size())

        block31 = self.relu(self.bn3(self.conv32(self.relu(self.bn3(self.conv31(block22))))))
        print('block31', block31.size())
        block32 = self.maxpool3(block31)
        print('block32', block32.size())

        block41 = self.relu(self.bn4(self.conv42(self.relu(self.bn4(self.conv41(block32))))))
        print('block41', block41.size())
        block42 = self.maxpool4(block41)
        print('block42', block42.size())

################################3
        print(block42.size())
        block51 = self.relu(self.bn5(self.conv52(self.relu(self.bn5(self.conv51(block42))))))
        print('block51', block51.size())
        block52 = self.transConv5(block51)
        print('block52', block52.size())

        print('cat', block41.size(), block52.size())
        # combined6 = torch.cat((self.center_crop(block41,block52),block52), dim=1)
        combined6 = torch.cat((block41,block52), dim=1)
        print('combined6',combined6.size())
        block61 = self.relu(self.bn6(self.conv62(self.relu(self.bn6(self.conv61(combined6))))))
        print('block61',block61.size())
        block62 = self.transConv6(block61)
        print('block62', block62.size())

        print('cat', block31.size(), block62.size())
        combined7 = torch.cat((block31, block62), dim=1)
        print('combined7',combined7.size())
        block71 = self.relu(self.bn7(self.conv72(self.relu(self.bn7(self.conv71(combined7))))))
        print('block71',block71.size())
        block72 = self.transConv7(block71)
        print('block72',block72.size())

        print('cat', block21.size(), block72.size())
        combined8 = torch.cat((block21, block72), dim=1)
        print('combined8',combined8.size())
        block81 = self.relu(self.bn8(self.conv82(self.relu(self.bn8(self.conv81(combined8))))))
        print('block81',block81.size())
        block82 = self.transConv8(block81)
        print('block82',block82.size())

        combined9 = torch.cat((block11, block82), dim=1)
        print('combined9',combined9.size())
        block91 = self.relu(self.bn9(self.conv92(self.relu(self.bn9(self.conv91(combined9))))))
        print('block91',block91.size())
        block92 = self.conv93(block91)
        print('block92',block92.size())

        # return torch.sigmoid(block92)
        return block92

    def center_crop(self, small, large):
        x1, y1 = large.size()[2], large.size()[3]
        x2, y2 = small.size()[2], small.size()[3]
        return large[:,:,int(x1/2 - x2/2) : int(x1/2 + x2/2), int(y1/2 - y2/2) : int(y1/2 + y2/2)]


