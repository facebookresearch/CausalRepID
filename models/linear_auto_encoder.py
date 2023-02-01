import torch
from torch import nn

class LinearAutoEncoder(torch.nn.Module):
    
    def __init__(self, data_dim, latent_dim, batch_norm= False):
        super(LinearAutoEncoder, self).__init__()        
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        if batch_norm:
            self.net= nn.Sequential(                        
                    nn.BatchNorm1d(self.data_dim),
                    nn.Linear(self.data_dim, self.latent_dim),            
                )
        else:
            self.net= nn.Sequential(                        
                    nn.Linear(self.data_dim, self.latent_dim),            
                )
        
    def forward(self, x):
        return self.net(x)
    