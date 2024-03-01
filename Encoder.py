import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1(x)))
        x2 = F.relu(self.bn(self.conv2(x)))
        x3 = F.relu(self.bn(self.conv3(x)))
        x4 = F.relu(self.bn(self.conv4(x)))
        x5 = F.relu(self.bn(self.conv5(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))))
        x = self.conv6(torch.cat((x1, x2, x3, x4, x5), dim=1))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.conv2_x = resnet50.layer1
        self.conv3_x = resnet50.layer2
        self.conv4_x = resnet50.layer3
        for i in range(6):
            self.conv4_x[i].conv1 = nn.Conv2d(512 if i == 0 else 1024, 256, kernel_size=1, stride=1,
                                              dilation=2, padding=2, bias=False)
            self.conv4_x[i].conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                                              dilation=2, padding=2, bias=False)
            self.conv4_x[i].conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1,
                                              dilation=2, padding=2, bias=False)
        self.conv5_x = resnet50.layer4
        for i in range(3):
            self.conv5_x[i].conv1 = nn.Conv2d(1024 if i == 0 else 2048, 512, kernel_size=1, stride=1,
                                              dilation=4, padding=4, bias=False)
            self.conv5_x[i].conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
                                              dilation=4, padding=4, bias=False)
            self.conv5_x[i].conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1,
                                              dilation=4, padding=4, bias=False)
        self.aspp = ASPP(in_channels=2048, out_channels=256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.aspp(x)
        return x
