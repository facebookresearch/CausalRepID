# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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

from models.encoder import Encoder
from models.poly_decoder import PolyDecoder

#Base Class
from algorithms.base_auto_encoder import AE

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helper import ValidationHelper
from utils.metrics import *

import wandb


class AE_Poly(AE):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, seed, device)        
        
        self.encoder= Encoder(self.args.data_dim, self.args.latent_dim).to(self.device)
        self.decoder= PolyDecoder(self.args.data_dim, self.args.latent_dim, self.args.poly_degree, self.device).to(self.device)                

        self.opt, self.scheduler= self.get_optimizer()
        
        if self.args.intervention_case:           
            self.res_dir= 'results/ae-poly/intervention/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
        else:
            self.res_dir= 'results/ae-poly/observation/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
                
        self.save_path= self.res_dir 
        
        if self.args.wandb_log:
            wandb.init(project="polynomial-identification", reinit=True)
            wandb.run.name=  'ae-poly/' + self.args.save_dir + 'seed_' + str(seed) + '/'