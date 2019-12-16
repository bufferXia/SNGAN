#这里使用的监督器 有点不一样，叫 Projection Discriminator
#可见 https://zhuanlan.zhihu.com/p/63353147 或者 论文原文
from torch import nn
from dis_models_pytorch.resblocks import FirstResBlockDiscriminator
from dis_models_pytorch.resblocks import ResBlockDiscriminator
from network_element import SpectralNorm
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):

    def __init__(self, n_class=17):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
                                    FirstResBlockDiscriminator(3,64),
                                   ResBlockDiscriminator(64,128),
                                   ResBlockDiscriminator(128,256),
                                   ResBlockDiscriminator(256, 512),
                                   ResBlockDiscriminator(512, 1024),
                                   ResBlockDiscriminator(1024, 1024),
                                   nn.ReLU()
        )

        self.linear = SpectralNorm(nn.Linear(1024, 1))
        self.embed = nn.Embedding(n_class, 1024)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = SpectralNorm(self.embed)



    def forward(self, input, class_id):
        out = self.model(input)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)

        output = self.linear(out)

        if class_id is not None:
            #  相当于不加 softmax 的 classifier, 直接提取 classifier 在 label 对应的维度的输出
            class_out = torch.sum(self.embed(class_id) * output, dim=1, keepdim=True)
            # 把两部分加起来作为 discriminator 的 output
            output = output + class_out

        # embed = self.embed(class_id)
        # prod = (out * embed).sum(1)

        return output