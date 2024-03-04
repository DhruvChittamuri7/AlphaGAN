import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=4, ndf=64, n_layers=3, patch_size=70):
        super(PatchGANDiscriminator, self).__init__()
        layers = [nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=patch_size, stride=1, padding=0)]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

