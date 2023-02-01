# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock
import sys

'''
Refernce for DeConv Blocks: https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
'''

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ImageDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()        
        
        self.latent_dim = latent_dim        
        self.width= 128
        self.num_Blocks=[2,2,2,2]
        self.in_planes = 512
        self.nc= 3
                
        self.linear =  [
                    nn.Linear(self.latent_dim, self.width),
                    nn.LeakyReLU(),
                    nn.Linear(self.width, 512),
        ]        
        self.linear= nn.Sequential(*self.linear)        

        self.layer4 = self._make_layer(BasicBlockDec, 256, self.num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, self.num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, self.num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, self.num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, self.nc, kernel_size=3, scale_factor=2)        
        
    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        return x
