import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
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
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        pretrained_dict = resnet50.state_dict()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1  # conv2_x
        self.layer2 = resnet50.layer2  # conv3_x

        self.layer3 = resnet50.layer3  # conv4_x
        for i in range(6):
            self.layer3[i].conv1 = nn.Conv2d(512 if i == 0 else 1024, 256, kernel_size=1, stride=1,
                                             dilation=2, padding=2, bias=False)
            self.layer3[i].conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             dilation=2, padding=2, bias=False)
            self.layer3[i].conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1,
                                             dilation=2, padding=2, bias=False)

        self.layer4 = resnet50.layer4  # conv5_x
        for i in range(3):
            self.layer4[i].conv1 = nn.Conv2d(1024 if i == 0 else 2048, 512, kernel_size=1, stride=1,
                                             dilation=4, padding=4, bias=False)
            self.layer4[i].conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             dilation=4, padding=4, bias=False)
            self.layer4[i].conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1,
                                             dilation=4, padding=4, bias=False)

        self.aspp = ASPP(2048, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
        state_dict.update(pretrained_dict)
        self.load_state_dict(state_dict)

    def forward(self, x):
        skip_rgb = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_conv1_x = x

        x, max_index = self.maxpool(x, return_indices=True)

        x = self.layer1(x)  # conv2_x
        skip_conv2_x = x

        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x
        x = self.aspp(x)
        return x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.skip_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv1_x = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.skip_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.deconv2_x = nn.Sequential(
            nn.Conv2d(64 + 32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.deconv3_x = nn.Sequential(
            nn.Conv2d(32 + 3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.deconv4_x = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index = x
        n, c, h, w = x.size()
        x = F.interpolate(size=(h * 2, w * 2), mode='bilinear', align_corners=True)

        skip_conv2_x = self.skip_1(skip_conv2_x)
        x = torch.cat([x, skip_conv2_x], dim=1)
        x = self.deconv1_x(x)

        x = self.unpooling(x, max_index)

        skip_conv1_x = self.skip_2(skip_conv1_x)
        x = torch.cat([x, skip_conv1_x], dim=1)
        x = self.deconv2_x(x)

        skip_rgb = skip_rgb[:, 0:3, :, :]
        x = torch.cat([x, skip_rgb], dim=1)
        x = self.deconv3_x(x)

        x = self.deconv4_x(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
