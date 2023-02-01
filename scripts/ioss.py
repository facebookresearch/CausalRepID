# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import math
import argparse
import numpy as np
import time
## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
from tqdm.notebook import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch.utils.data as data

from scipy.stats import ortho_group
from utils.metrics import get_cross_correlation
from scipy.stats import bernoulli

def get_predictions(model, train_dataset, device= None):
    
    
    true_z= []
    pred_z= []
    count= 0
    
    for batch_idx, (data_inputs, data_latents), in enumerate(train_dataset):
        with torch.no_grad():
            data_inputs = data_inputs.to(device)
            preds = model(data_inputs)
            
            true_z.append(data_latents)
            pred_z.append(preds)
        
            count+=1
                

    true_z= torch.cat(true_z).detach().numpy()
    pred_z= torch.cat(pred_z).cpu().detach().numpy()
    
    return true_z, pred_z

class LinearEncoder(nn.Module):
    
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        
    def forward(self, x):
        x = self.linear(x)        
        return x

class data_class_loader(data.Dataset):
    
    def __init__(self,  dataset):
        
        super().__init__()
        self.dataset_generate(dataset)
        self.size = dataset[0].size()[0]
        
    def dataset_generate(self, dataset):
        
        self.data  = dataset[0]
        self.latent = dataset[1]
        self.label = dataset[2].T[0].to(int)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_latent = self.latent[idx]
        data_label = self.label[idx]
        return data_point, data_latent

def IOSS(mu, n_draws=10000, robust_k_prop = 0.01):
    stdmu = (mu - torch.min(mu,dim=0)[0])/ (torch.max(mu,dim=0)[0]-torch.min(mu,dim=0)[0])
    
    K = np.int(robust_k_prop * mu.shape[0]) + 1

    maxs = torch.topk(stdmu, K, dim=0)[0][-1,:]
    mins = -(torch.topk(-stdmu, K, dim=0)[0][-1,:])    

    smps = (torch.stack([torch.rand(n_draws).cuda() * (maxs[i]-mins[i]) + mins[i] for i in range(stdmu.shape[1])], dim=1))
    
    min_dist = (torch.min(torch.cdist(smps, stdmu.cuda()), dim=1)[0])
    # ortho = (torch.mean(min_dist,dim=0))
    ortho = (torch.topk(min_dist, np.int(robust_k_prop*n_draws)+1, dim=0))[0][-1]
    # ortho = torch.max(min_dist,dim=0)[0]
    return ortho


def loss_intervention(decoder, preds, inputs, ioss_penalty=0.1, device=[]):
    
#     constraint= torch.mean( IOSS(preds) )
    constraint= torch.tensor(0.0).to(device)
    total_pairs= 3
    for idx in range(total_pairs):
        perm= torch.randperm(preds.shape[1])
        constraint+= torch.mean(IOSS( preds[:, perm[:2]] ))
    
    #Reconstruction Loss
    criterion = nn.MSELoss()
    preds= decoder(preds)
    loss = criterion(preds,inputs)
    
    #Final Loss
    theta = ioss_penalty
    loss = loss + theta*constraint
    return loss  

def train_model(encoder, decoder, optimizer, data_loader, num_epochs, ioss_penalty= 0.1):
    # Set model to train mode
    encoder.train()
    decoder.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        
        #MCC
        z, z_pred= get_predictions(encoder, data_loader, device= device)
        mcc= get_cross_correlation({'te':z}, {'te':z_pred})
        print('MCC: ', mcc)
        
        print('Epoch: ', epoch)
        train_loss= 0.0
        batch_count= 0
        for data_inputs, data_latents, in data_loader:
            data_inputs = data_inputs.to(device)
            preds = encoder(data_inputs)
            
#             preds = preds.squeeze(dim=1) 
#             loss = loss_intervention(preds, data_inputs)
            loss = loss_intervention(decoder, preds, data_inputs, ioss_penalty= ioss_penalty, device= device)
            train_loss+= loss.item()
            batch_count+=1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Loss: ', train_loss/batch_count)        

        
# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--total_samples', type=int, default=50000,
                    help='')
parser.add_argument('--latent_dim', type=int, default= 10,
                    help='')
parser.add_argument('--latent_case', type=str, default='uniform',
                    help='uniform, uniform_corr')
parser.add_argument('--ioss_penalty', type=float, default= 10.0,
                    help='')
parser.add_argument('--batch_size', type=int, default= 16,
                    help='')
parser.add_argument('--lr', type=float, default= 0.001,
                    help='')
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_epochs', type=int, default= 300,
                    help='')
parser.add_argument('--seed', type=int, default=3,
                    help='')


args = parser.parse_args()
n = args.total_samples
d = args.latent_dim
latent_case= args.latent_case
ioss_penalty= args.ioss_penalty
lr= args.lr
weight_decay= args.weight_decay

# A = np.random.uniform(size=(d,d))
A = ortho_group.rvs(d)

# Observational data 
if latent_case == 'uniform':
    Zobs = np.random.uniform(size=(n,d))
elif latent_case == 'uniform_corr':    
    Zobs= np.zeros((n, d))
    for d_idx in range(0 , d, 2):
        print('Latent entries for the pair: ', d_idx, d_idx + 1)
        p1= bernoulli.rvs(0.5, size=n)
        p2= bernoulli.rvs(0.9, size=n)
        z_11= np.random.uniform(low=0, high=5, size=n)
        z_12= np.random.uniform(low=-5, high=0, size=n)
        z_21= np.random.uniform(low=0, high=3, size=n)
        z_22= np.random.uniform(low=-3, high=0, size=n)

        for idx in range(n):
            if p1[idx] == 1:
                Zobs[idx, d_idx + 0]= z_11[idx]
                if p2[idx] == 1:
                    Zobs[idx, d_idx + 1]= z_21[idx]
                else:
                    Zobs[idx, d_idx + 1]= z_22[idx]
            else:
                Zobs[idx, d_idx + 0]= z_12[idx]
                if p2[idx] == 1:
                    Zobs[idx, d_idx + 1]= z_22[idx]
                else:
                    Zobs[idx, d_idx + 1]= z_21[idx]
        
# print(100*np.sum(Zobs[:,0]*Zobs[:,1]<0)/n)

Xobs = np.matmul(Zobs,A)
Eobs = np.zeros((n,1))

X = Xobs
Z = Zobs
E = Eobs
X = torch.tensor(X, dtype=torch.float32)
E = torch.tensor(E, dtype=torch.float32)
Data_train = (X, Z, E)


# data prep
data_train_obj = data_class_loader(Data_train)
train_data_loader = data.DataLoader(data_train_obj, batch_size=32, shuffle=True)
# device = torch.device("cuda:0")
device = torch.device("cuda:0")

# model and optimizer init
num_inputs= Data_train[0].size()[1]
encoder = LinearEncoder(num_inputs=num_inputs, num_outputs=num_inputs)
encoder.to(device)
decoder = LinearEncoder(num_inputs=num_inputs, num_outputs=num_inputs)
decoder.to(device)
optimizer= torch.optim.Adam([
                    {'params': filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters()) )}, 
                    ], lr= lr, weight_decay= weight_decay )
num_epochs = 200

# train model 
train_model(encoder, decoder, optimizer, train_data_loader, num_epochs, ioss_penalty= ioss_penalty)
