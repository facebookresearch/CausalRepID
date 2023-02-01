# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#Common imports
import sys
import os
import argparse
import random
import copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA

from algorithms.base_auto_encoder import AE
from algorithms.poly_auto_encoder import AE_Poly
from algorithms.ioss_auto_encoder import AE_IOSS
from algorithms.image_auto_encoder import AE_Image
from utils.metrics import *
from utils.helper import *


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--method_type', type=str, default='ae_poly',
                   help= 'ae; ae_poly; ae_image')
parser.add_argument('--latent_case', type=str, default='uniform',
                    help='laplace; uniform')
parser.add_argument('--data_dim', type=int, default= 200,
                    help='')
parser.add_argument('--latent_dim', type=int, default= 10,
                    help='')
parser.add_argument('--poly_degree', type=int, default= 2,
                    help='')
parser.add_argument('--batch_size', type=int, default= 16,
                    help='')
parser.add_argument('--lr', type=float, default= 1e-3,
                    help='')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_epochs', type=int, default= 200,
                    help='')
parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--intervention_case', type= int, default= 0, 
                   help= '')
parser.add_argument('--train_base_model', type=int, default=1,
                   help='Train the base auto encoder')
parser.add_argument('--train_ioss_transformation', type=int, default=0,
                   help='Learn the IOSS transformation from the base model representations')
parser.add_argument('--wandb_log', type=int, default=0,
                   help='')
parser.add_argument('--cuda_device', type=int, default=-1, 
                    help='Select the cuda device by id among the avaliable devices' )

args = parser.parse_args()
method_type= args.method_type
latent_case= args.latent_case
data_dim= args.data_dim
latent_dim= args.latent_dim
poly_degree= args.poly_degree
batch_size= args.batch_size
lr= args.lr
weight_decay= args.weight_decay
num_epochs= args.num_epochs
seed= args.seed
intervention_case= args.intervention_case
train_base_model= args.train_base_model
train_ioss_transformation= args.train_ioss_transformation
wandb_log= args.wandb_log
cuda_device= args.cuda_device

if 'balls' in latent_case:
    save_dir= latent_case + '/'
else:
    save_dir= 'polynomial' + '_latent_' + latent_case + '_poly_degree_' + str(poly_degree) + '_data_dim_' + str(data_dim) + '_latent_dim_' + str(latent_dim)  + '/'

args.save_dir= save_dir

#GPU
if cuda_device == -1:
    device= torch.device("cpu")
else:
    device= torch.device("cuda:" + str(cuda_device))
    
if device:
    kwargs = {'num_workers': 0, 'pin_memory': False} 
else:
    kwargs= {}

#Seed values
random.seed(seed*10)
np.random.seed(seed*10) 
torch.manual_seed(seed*10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed*10)
        
    
# Load Dataset
train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, observation_case=1, intervention_case= intervention_case, latent_case= latent_case, seed= seed, kwargs=kwargs)

#Load Algorithm
if method_type == 'ae':
    method= AE(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)
elif method_type == 'ae_poly':
    method= AE_Poly(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)
elif method_type == 'ae_image':
    method= AE_Image(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)    
else:
    print('Error: Incorrect method type')
    sys.exit(-1)
    
# Training
if train_base_model:
    method.train()
    
#Train with IOSS Loss
if train_ioss_transformation:
    method.load_model()

    #Sample data from only the observational distribution
    train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, seed= seed, observation_case=1, intervention_case= 0, kwargs=kwargs)
    
    #Obtain Predictions and Reconstruction Loss
    res= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device)
    
    #Sample dataloaders for finetuning the representations
    train_dataset, val_dataset, test_dataset= sample_finetune_data_loaders(res['pred_z'], res['true_z'], save_dir, batch_size, kwargs= kwargs)
    
    ioss_method= AE_IOSS(args, train_dataset, val_dataset, test_dataset, seed=seed, device=device, base_algo= method_type)
    ioss_method.train()  
