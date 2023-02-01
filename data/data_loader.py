# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class BaseDataLoader(data_utils.Dataset):
    def __init__(self, data_dir='', data_case='train', seed= 0, observation_case= True, intervention_case=False):
        self.data_case= data_case
        self.seed= seed
        self.obs_data_dir = 'data/datasets/' + 'seed_' + str(seed) + '/observation/' + data_dir
        self.itv_data_dir = 'data/datasets/' + 'seed_' + str(seed) + '/intervention/' + data_dir
        self.observation_case= observation_case
        self.intervention_case= intervention_case
        
        self.data, self.latents, self.intervention_indices= self.load_data(self.data_case)
        
    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        z = self.latents[index]
        y= self.intervention_indices[index]
            
        return x, z, y
    
    def load_data(self, data_case):
        
        x_obs= np.load(self.obs_data_dir + data_case +  '_' + 'x' + '.npy')
        z_obs= np.load(self.obs_data_dir + data_case +  '_' + 'z' + '.npy')            
        y_obs= np.load(self.obs_data_dir + data_case +  '_' + 'y' + '.npy')            

        x_itv= np.load(self.itv_data_dir + data_case +  '_' + 'x' + '.npy')
        z_itv= np.load(self.itv_data_dir + data_case +  '_' + 'z' + '.npy')            
        y_itv= np.load(self.itv_data_dir + data_case +  '_' + 'y' + '.npy')            
        
        if self.observation_case and self.intervention_case:
            x= np.concatenate((x_obs, x_itv), axis=0)
            z= np.concatenate((z_obs, z_itv), axis=0)
            y= np.concatenate((y_obs, y_itv), axis=0)
            
        elif self.observation_case:
            x= x_obs
            z= z_obs
            y= y_obs
        
        elif self.intervention_case:
            x= x_itv
            y= y_itv
            z= z_itv
            
        x= torch.tensor(x).float()
        z= torch.tensor(z).float()
        y= torch.tensor(y).float()
        
        return x, z, y