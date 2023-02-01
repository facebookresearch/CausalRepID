import os
import sys
import numpy as np

import torch
import torch.utils.data as data_utils

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from data.data_loader import BaseDataLoader
from data.fine_tune_loader import FineTuneDataLoader
from data.balls_dataset_loader import BallsDataLoader


class ValidationHelper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = -1
        
    def save_model(self, validation_loss, epoch):
        if validation_loss < (self.min_validation_loss + self.min_delta):
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch
            self.counter= 0
            return True
        return False

    def early_stop(self, validation_loss):        
        if validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
def sample_base_data_loaders(data_dir, batch_size, observation_case= 1, intervention_case= 0,  latent_case='', seed=0, kwargs={}):
        
    if 'balls' in latent_case:
        
        data_obj= BallsDataLoader(data_dir= data_dir, data_case='train', observation_case = observation_case, intervention_case= intervention_case)
        train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

        data_obj= BallsDataLoader(data_dir= data_dir, data_case='val', observation_case = observation_case, intervention_case= intervention_case)
        val_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

        data_obj= BallsDataLoader(data_dir= data_dir, data_case='test', observation_case = observation_case, intervention_case= intervention_case)
        test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )
    
    else:
        
        data_obj= BaseDataLoader(data_dir= data_dir, data_case='train', seed= seed, observation_case = observation_case, intervention_case= intervention_case)
        train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

        data_obj= BaseDataLoader(data_dir= data_dir, data_case='val', seed= seed, observation_case = observation_case, intervention_case= intervention_case)
        val_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

        data_obj= BaseDataLoader(data_dir= data_dir, data_case='test', seed= seed, observation_case = observation_case, intervention_case= intervention_case)
        test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )
    
    return train_dataset, val_dataset, test_dataset


def sample_finetune_data_loaders(pred_z, true_z, data_dir, batch_size, kwargs= {}):
        
    data_obj= FineTuneDataLoader(pred_z, true_z, data_dir= data_dir, data_case='train')
    train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

    data_obj= FineTuneDataLoader(pred_z, true_z, data_dir= data_dir, data_case='val')
    val_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

    data_obj= FineTuneDataLoader(pred_z, true_z, data_dir= data_dir, data_case='test')
    test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )
    
    return train_dataset, val_dataset, test_dataset