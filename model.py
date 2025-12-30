import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(width, width, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.head  = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.head(x)
        x = F.interpolate(x, size=(28, 28), mode="nearest")
        return x

