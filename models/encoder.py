# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

class Encoder(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim):
        super(Encoder, self).__init__()        
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim= 100
        
        self.net= nn.Sequential(
                        
                    nn.Linear(self.data_dim, self.hidden_dim),
                    nn.LeakyReLU(0.5),
                    
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(0.5),

                    nn.Linear(self.hidden_dim, self.latent_dim),            
                )
        
    def forward(self, x):
        return self.net(x)