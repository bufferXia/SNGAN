import torch
from gen_models_pytorch.resblocks import ConvBlock
from torch.nn import functional as F
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, code_dim=128, n_class=17):
        super(Generator,self).__init__()

        self.lin_code = nn.Linear(code_dim, 4 * 4 * 1024)
        self.conv = nn.ModuleList([ConvBlock(1024, 1024, n_class=n_class),
                                   ConvBlock(1024, 512, n_class=n_class),
                                   ConvBlock(512, 256, n_class=n_class),
                                   ConvBlock(256, 128, n_class=n_class),
                                   ConvBlock(128, 64, n_class=n_class)])

        self.bn = nn.BatchNorm2d(64)
        self.colorize = nn.Conv2d(64, 3, [3, 3], padding=1)

    def forward(self, input, class_id):
        out = self.lin_code(input)
        out = out.view(-1, 1024, 4, 4)

        for conv in self.conv:
            if isinstance(conv, ConvBlock):
                out = conv(out, class_id)
            else:
                out = conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.colorize(out)

        return F.tanh(out)