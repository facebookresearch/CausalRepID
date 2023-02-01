# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class Decoder(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim):
        super(Decoder, self).__init__()        
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim= 200
        
        self.net= nn.Sequential(
            
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.LeakyReLU(0.5),
                    
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(0.5),
            
                    nn.Linear(self.hidden_dim, self.data_dim),
            
                )
        
    def forward(self, z):
        return self.net(z)