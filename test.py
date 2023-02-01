#Common imports
import sys
import os
import argparse
import random
import copy

import torch
import torch.utils.data as data_utils
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
                   help= 'ae, ae_poly')
parser.add_argument('--latent_case', type=str, default='uniform',
                    help='laplace; uniform')
parser.add_argument('--data_dim', type=int, default= 200,
                    help='')
parser.add_argument('--latent_dim', type=int, default= 6,
                    help='')
parser.add_argument('--poly_degree', type=int, default= 2,
                    help='')
parser.add_argument('--batch_size', type=int, default= 16,
                    help='')
parser.add_argument('--lr', type=float, default= 1e-3,
                    help='')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_seeds', type=int, default=5,
                    help='')
parser.add_argument('--intervention_case', type= int, default= 0, 
                   help= '')
parser.add_argument('--eval_ioss_transformation', type=int, default=0,
                   help='Evaluate the IOSS transformation from the base model representation')
parser.add_argument('--eval_intervene_transformation', type=int, default=0,
                   help='Evaluate the Intervention transformation from the base model representation')
parser.add_argument('--eval_dgp', type=int, default= 0,
                    help= 'Evaluate the function from z -> x and x -> z in the true DGP')
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
num_seeds= args.num_seeds
intervention_case= args.intervention_case
eval_dgp= args.eval_dgp
eval_ioss_transformation= args.eval_ioss_transformation
eval_intervene_transformation= args.eval_intervene_transformation
wandb_log= args.wandb_log
cuda_device= args.cuda_device

if 'balls' in latent_case:
    save_dir= latent_case + '/'
else:
    save_dir= 'polynomial' + '_latent_' + latent_case + '_poly_degree_' + str(poly_degree) + '_data_dim_' + str(data_dim) + '_latent_dim_' + str(latent_dim)  + '/'
args.save_dir= save_dir
    
#GPU
#GPU
if cuda_device == -1:
    device= "cpu"
else:
    device= torch.device("cuda:" + str(cuda_device))
    
if device:
    kwargs = {'num_workers': 0, 'pin_memory': False} 
else:
    kwargs= {}

res={}

for seed in range(num_seeds):
        
    #Seed values
    random.seed(seed*10)
    np.random.seed(seed*10)
    torch.manual_seed(seed*10)
    
    # Load Dataset
    train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size,  seed= seed, observation_case=1, intervention_case= intervention_case, latent_case= latent_case, kwargs=kwargs)
    
    #Load Algorithm
    if method_type == 'ae':
        method= AE(args, train_dataset, val_dataset, test_dataset, seed=seed)
    elif method_type == 'ae_poly':
        method= AE_Poly(args, train_dataset, val_dataset, test_dataset, seed=seed)
    elif method_type == 'ae_image':
        method= AE_Image(args, train_dataset, val_dataset, test_dataset, seed=seed, device= device)    
    
    
    # Evaluate the models learnt on true latent variables
    if eval_dgp:    
        # X->Z prediction R2
        x, z= get_predictions_check(train_dataset, test_dataset)
        rmse, r2= get_indirect_prediction_error(x, z)

        key= 'oracle_pred_rmse'
        if key not in res.keys():
            res[key]= []
        res[key].append(rmse)

        key= 'oracle_pred_r2'
        if key not in res.keys():
            res[key]= []
        res[key].append(r2)


        # Z->X prediction R2
        x, z= get_predictions_check(train_dataset, test_dataset)
        rmse, r2= get_indirect_prediction_error(z, x)

        key= 'debug_pred_rmse'
        if key not in res.keys():
            res[key]= []
        res[key].append(rmse)

        key= 'debug_pred_r2'
        if key not in res.keys():
            res[key]= []
        res[key].append(r2)      

        
    # Evaluate the base model
    method.load_model()        
#     method.load_intermediate_model(epoch=10)        
    
    #Latent Prediction Error
    rmse,r2= method.eval_identification()

    key= 'latent_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(rmse)

    key= 'latent_pred_r2'
    if key not in res.keys():
        res[key]=[]
    res[key].append(r2)
  

    # Evaluating MCC on the observational data with representations from Step 1    
    
    #Sample data from only the observational distribution
    train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, seed= seed, observation_case=1, intervention_case= 0, latent_case= latent_case, kwargs=kwargs)
    
    #Obtain Predictions and Reconstruction Loss
    logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, plot= False)
    
    #Prediction RMSE
    key= 'recon_rmse'
    if key not in res.keys():
        res[key]= []
    res[key].append(logs['recon_loss']['val'])    
    
    print('RMSE Val: ', logs['recon_loss']['val'])    
    
    #MCC
    if 'balls' not in latent_case:
        mcc= get_cross_correlation(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']))
        key= 'mcc'
        if key not in res.keys():
            res[key]= []
        for item in mcc:
            res[key].append(item)
    
    
    if eval_intervene_transformation:
    
        #Sample only from interventional distribution
        train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, seed= seed, latent_case= latent_case, observation_case=0, intervention_case= 1, kwargs=kwargs)

        #Obtain Predictions and Reconstruction Loss
        logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, plot= False)

        # Intervention Specific Metric
        if 'balls' not in latent_case:
            reg_models= intervene_metric(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), model_train=1)
        else:
            reg_models= intervene_metric_image(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), copy.deepcopy(logs['true_y']), model_train=1, model= 'mlp')
            

        #Sample data from only the observational distribution
        train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, seed= seed, latent_case= latent_case, observation_case=1, intervention_case= 0, kwargs=kwargs)

        #Obtain Predictions and Reconstruction Loss
        logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, plot= False)
        
        # Intervention Specific Metric
        if 'balls' not in latent_case:
            logs['pred_z']= intervene_metric(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), model_train=0, list_models=reg_models)
        else:
            logs['pred_z'], logs['true_z']= intervene_metric_image(copy.deepcopy(logs['pred_z']), copy.deepcopy(logs['true_z']), copy.deepcopy(logs['true_y']), model_train=0, list_models= reg_models,  model= 'mlp')
            
    
    #Sample dataloaders for finetuning the representations
    if eval_ioss_transformation:
        
        #Sample data from only the observational distribution
        train_dataset, val_dataset, test_dataset= sample_base_data_loaders(save_dir, batch_size, seed= seed,  latent_case= latent_case, observation_case=1, intervention_case= 0, kwargs=kwargs)

        #Obtain Predictions and Reconstruction Loss
        logs= get_predictions(method.encoder, method.decoder, train_dataset, val_dataset, test_dataset, device=method.device, plot= False)
        
        #Train with IOSS Loss
        train_dataset, val_dataset, test_dataset= sample_finetune_data_loaders(logs['pred_z'], logs['true_z'], save_dir, batch_size, kwargs= kwargs)
        
        ioss_method= AE_IOSS(args, train_dataset, val_dataset, test_dataset, seed=seed, device=device, base_algo= method_type)
        ioss_method.load_model()

        #Obtain Predictions and Reconstruction Loss
        logs= get_predictions(ioss_method.encoder, ioss_method.decoder, ioss_method.train_dataset, ioss_method.val_dataset, ioss_method.test_dataset, device=ioss_method.device, plot= False)

    #MCC
    if eval_ioss_transformation or eval_intervene_transformation:
        mcc= get_cross_correlation(logs['pred_z'], logs['true_z'])
        print('MCC: ', mcc)
        key= 'mcc_tune'
        if key not in res.keys():
            res[key]= []
        for item in mcc:
            res[key].append(item)
    
print('Final Results')
print(res.keys())

for key in res.keys():
    res[key]= np.array(res[key])
    print('Metric: ', key, np.mean(res[key]), np.std(res[key])/np.sqrt(num_seeds))
            