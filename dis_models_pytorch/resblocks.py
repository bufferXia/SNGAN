# 输入层的残差块和中间的残差块会不太一样，
# 中间的res 要先经过一个激活函数
# 但是 输入层的 resblock 开始不需要先经过一个激活函数

import numpy as np
from torch import nn
from network_element import SpectralNorm

class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )

        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels,downsample=False):
        super(ResBlockDiscriminator, self).__init__()

        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        if self.learnable_sc:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if self.learnable_sc:
            nn.init.xavier_uniform(self.conv3.weight.data, 1.)

        self.model = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2))

        bypass_cache = []

        if self.learnable_sc:
            bypass_cache.append(SpectralNorm(self.conv3))
            if self.downsample:
                bypass_cache.append(nn.AvgPool2d(2))

        self.bypass = nn.Sequential(*bypass_cache)

    def forward(self, x):
        return self.model(x) + self.bypass(x)