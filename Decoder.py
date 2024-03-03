import torch as t
from torch import nn
import torch.nn.functional as F


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
        x = t.cat([x, skip_conv2_x], dim=1)
        x = self.deconv1_x(x)

        x = self.unpooling(x, max_index)

        skip_conv1_x = self.skip_2(skip_conv1_x)
        x = t.cat([x, skip_conv1_x], dim=1)
        x = self.deconv2_x(x)

        skip_rgb = skip_rgb[:, 0:3, :, :]
        x = t.cat([x, skip_rgb], dim=1)
        x = self.deconv3_x(x)

        x = self.deconv4_x(x)
        return x
