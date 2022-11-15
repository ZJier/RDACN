# Author: Junjie Zhang
import torch.nn as nn
import torch
from torch.nn import init
import numpy as np


class mish(nn.Module):
    def __init__(self, inplace=False):
        super(mish, self).__init__()
        self.inplace = inplace
    def forward(self, input):
        return input * torch.tanh(F.softplus(input))


class Asym_trans(nn.Module):

    def __init__(self, in_planes, out_planes, group=3):
        super(Asym_trans, self).__init__()
        self.group = group
        self.act = mish()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(self.act(self.bn(x)))
        return out


class Asymmblock(nn.Module):

    def __init__(self, in_planes, mid_planes):
        super(Asymmblock, self).__init__()
        self.act = mish()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        mid_planes = in_planes + 2*mid_planes

        self.bn2_1 = nn.BatchNorm2d(mid_planes)
        self.conv2_1 = nn.Conv2d(mid_planes, mid_planes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2_2 = nn.BatchNorm2d(mid_planes)
        self.conv2_2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=(3, 1), padding=(1, 0), bias=False)

        self.bn3 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, in_planes, kernel_size=1, bias=False)

    def forward(self, x):
        output1 = self.conv1(self.act(self.bn1(x)))
        output11 = torch.cat([output1, x, output1], 1)
        output2_1 = self.conv2_1(self.act(self.bn2_1(output11)))
        output2_2 = self.conv2_2(self.act(self.bn2_2(output2_1)))
        output3 = self.conv3(self.act(self.bn3(output2_2)))
        output = output3 + x
        return output


class RDACN(nn.Module):
    def __init__(self, in_channels, num_classes, mid_planes=36, growthrate=1.5):
        super(RDACN, self).__init__()
        num_planes = 2*mid_planes
        self.mid_planes = mid_planes
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.multiblock1 = Asymmblock(num_planes, mid_planes)
        out_planes = int(math.floor(num_planes * growthrate))
        self.growth1 = Asym_trans(num_planes, out_planes)
        num_planes = out_planes

        self.multiblock3 = Asymmblock(num_planes, mid_planes)

        self.bn = nn.BatchNorm2d(num_planes)
        self.act = mish()
        self.linear = nn.Linear(num_planes, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = np.squeeze(x, axis=1)
        out = self.conv1(out)
        out = self.growth1(self.multiblock1(out))
        out = self.multiblock3(out)
        out = self.pooling(self.act(self.bn(out)))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
