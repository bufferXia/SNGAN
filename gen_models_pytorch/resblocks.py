import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from network_element import ConditionalNorm

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=True,
                 activation=F.relu, upsample=True,skip=True):
        super(ConvBlock,self).__init__()
        #在snGan的原文(原文代码)中，生成器是不需要添加谱正则化的，因为 李普希兹约束是对监督器添加的
        self.conv1 = nn.Conv2d(in_channel, out_channel,kernel_size, stride, padding,bias=False if bn else True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,kernel_size, stride, padding,bias=False if bn else True)
        self.conv_skip = nn.Conv2d(in_channel, out_channel,1, 1, 0)


        self.skip_proj = False  #当残差块的输入和输出通道不相等，或者输入输出尺寸不想等，添加一个卷积层来匹配一下
        if in_channel != out_channel or upsample :
            self.conv_skip = nn.Conv2d(in_channel, out_channel,
                                                     1, 1, 0)
            self.skip_proj = True

        self.upsample = upsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.norm1 = ConditionalNorm(in_channel, n_class)
            self.norm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, input, class_id=None):
        out = input

        if self.bn:
            out = self.norm1(out, class_id)
        out = self.activation(out)
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
        out = self.conv1(out)
        if self.bn:
            out = self.norm2(out, class_id)
        out = self.activation(out)
        out = self.conv2(out)


        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_skip(skip)


        else:
            skip = input

        return out + skip