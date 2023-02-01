import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class ImageDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()        
        
        self.latent_dim = latent_dim        
        self.width= 128
        self.nc= 3
                
        self.linear =  [
                    nn.Linear(self.latent_dim, self.width),
                    nn.LeakyReLU(),
                    nn.Linear(self.width, 1024),
                    nn.LeakyReLU(),
        ]        
        self.linear= nn.Sequential(*self.linear)        

        self.conv= [
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),           
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, self.nc, 4, stride=2, padding=1),
        ]
        self.conv= nn.Sequential(*self.conv)


    def forward(self, z):
        x = self.linear(z)        
        x = x.view(z.size(0), 64, 4, 4)
        x = self.conv(x)
        return x
