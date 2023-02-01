import os
import sys
import math

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from utils.metrics import *

from models.image_encoder import ImageEncoder as Encoder
from models.image_decoder import ImageDecoder as Decoder
# from models.image_resnet_decoder import ImageDecoder as Decoder

#Base Class
from algorithms.base_auto_encoder import AE

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helper import ValidationHelper
from utils.metrics import *

import wandb


class AE_Image(AE):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, seed, device)        
        
        self.encoder= Encoder(self.args.latent_dim).to(self.device)
        self.decoder= Decoder(self.args.latent_dim).to(self.device)

        self.opt, self.scheduler= self.get_optimizer()        
        self.validation_helper= ValidationHelper(patience=100)                
        
        if self.args.intervention_case:           
            self.res_dir= 'results/ae-image/intervention/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
        else:
            self.res_dir= 'results/ae-image/observation/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
                
        self.save_path= self.res_dir 
        
        if self.args.wandb_log:
            wandb.init(project="image-dataset-identification", reinit=True)
            wandb.run.name=  'ae-image/' + self.args.save_dir + 'seed_' + str(seed) + '/'
            
    def save_intermediate_model(self, epoch= -1):
        
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        
        torch.save(self.encoder.state_dict(), self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_epoch_' + str(epoch) +  '_encoder.pth')        
        
        torch.save(self.decoder.state_dict(), self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_epoch_' + str(epoch) + '_decoder.pth')        
        return
    
    def load_intermediate_model(self, epoch):
        
        self.encoder.load_state_dict(torch.load(self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size)  + '_epoch_' + str(epoch) + '_encoder.pth'))
        self.encoder.eval()
        
        self.decoder.load_state_dict(torch.load(self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size)  + '_epoch_' + str(epoch) + '_decoder.pth'))
        self.decoder.eval()
        return        
