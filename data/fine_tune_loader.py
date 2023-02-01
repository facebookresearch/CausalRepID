# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

#Base Class
from data.data_loader import BaseDataLoader

class FineTuneDataLoader(BaseDataLoader):
    def __init__(self, pred_z, true_z, data_dir='', data_case='train'):
        
        super().__init__(data_dir= data_dir, data_case= data_case)        
        
        self.data, self.latents= self.load_data_updated(pred_z, true_z, self.data_case)
                            
    def load_data_updated(self, pred_z, true_z, data_case):
        if data_case == 'train':
            x= pred_z['tr']
            z= true_z['tr']
        elif data_case == 'val':
            x= pred_z['val']
            z= true_z['val']
        elif data_case == 'test':
            x= pred_z['te']
            z= true_z['te']
        
        scaler = StandardScaler()
        scaler.fit(x)
        x= scaler.transform(x)

        x= torch.tensor(x).float()
        z= torch.tensor(z).float()
        
        return x, z