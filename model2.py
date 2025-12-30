import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MyModel(nn.Module):

    def __init__(self, width=48):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3, width)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(width, width * 2)
        self.down2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(width * 2, width * 2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(width * 2 + width * 2, width)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(width + width, width)

        # Head
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        y = self.head(d2)
        # y = F.interpolate(y, size=(28, 28), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(28, 28), mode="nearest")
        return y