import torch as t
from torch import nn
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
        x = self.conv6(t.cat((x1, x2, x3, x4, x5), dim=1))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dilation != 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        downsample_stride = stride if dilation == 1 else 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=downsample_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_rgb = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_conv1_x = x
        x, max_index = self.maxpool(x)
        x = self.layer1(x)
        skip_conv2_x = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        pretrained_resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        pretrained_dict = pretrained_resnet50.state_dict()
        atrous_resnet_dict = self.resnet50.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in atrous_resnet_dict}
        atrous_resnet_dict.update(pretrained_dict)
        self.resnet50.load_state_dict(atrous_resnet_dict)

        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index = self.resnet50(x)
        x = self.aspp(x)
        return x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.bilinear = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )

        self.skip_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2)

        )

        self.deconv1_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256 + 48, out_channels=256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.skip_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.deconv2_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 + 32, out_channels=64, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.deconv3_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 + 4, out_channels=32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skip_rgb, skip_conv1_x, skip_conv2_x, max_index = x
        x = self.bilinear(x)
        skip_conv2_x = self.skip_2(skip_conv2_x)
        x = t.cat([x, skip_conv2_x], dim=1)
        x = self.deconv1_x(x)
        x = self.unpooling(x, max_index)
        skip_conv1_x = self.skip_1(skip_conv1_x)
        x = t.cat([x, skip_conv1_x], dim=1)
        x = self.deconv2_x(x)
        x = t.cat([x, skip_rgb], dim=1)
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
