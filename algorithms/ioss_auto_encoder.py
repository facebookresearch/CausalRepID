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

from models.linear_auto_encoder import LinearAutoEncoder

#Base Class
from algorithms.base_auto_encoder import AE

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helper import ValidationHelper
from utils.metrics import *

def IOSS(mu, n_draws=10000, robust_k_prop = 0.01, device= None):
    stdmu = (mu - torch.min(mu,dim=0)[0])/ (torch.max(mu,dim=0)[0]-torch.min(mu,dim=0)[0])

    K = np.int(robust_k_prop * mu.shape[0]) + 1

    maxs = torch.topk(stdmu, K, dim=0)[0][-1,:]
    mins = -(torch.topk(-stdmu, K, dim=0)[0][-1,:])    

    smps = (torch.stack([torch.rand(n_draws).to(device) * (maxs[i]-mins[i]) + mins[i] for i in range(stdmu.shape[1])], dim=1))
    min_dist = (torch.min(torch.cdist(smps, stdmu.to(device)), dim=1)[0])
    # ortho = (torch.mean(min_dist,dim=0))
    ortho = (torch.topk(min_dist, np.int(robust_k_prop*n_draws)+1, dim=0))[0][-1]
    # ortho = torch.max(min_dist,dim=0)[0]
    return ortho


class AE_IOSS(AE):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None, base_algo= 'ae_poly'):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, seed, device)  
                
        self.encoder= LinearAutoEncoder(self.args.latent_dim, self.args.latent_dim, batch_norm= 0).to(self.device)
        self.decoder= LinearAutoEncoder(self.args.latent_dim, self.args.latent_dim, batch_norm= 0).to(self.device)                
        
        self.opt, self.scheduler= self.get_optimizer()                
        self.base_algo = base_algo
        
        if self.args.intervention_case:           
            self.res_dir= 'results/ae-ioss/intervention/' + self.base_algo + '/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
        else:
            self.res_dir= 'results/ae-ioss/observation/' + self.base_algo + '/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
                
        self.save_path= self.res_dir
            
    def compute_loss(self, z_pred, x_pred, x):
        
        loss= torch.mean(((x-x_pred)**2))

        total_pairs= 3
        lambda_reg= 10.0        
        ioss_penalty= torch.tensor(0.0).to(self.device)
        for idx in range(total_pairs):
            perm= torch.randperm(z_pred.shape[1])
            ioss_penalty+= torch.mean(IOSS( z_pred[:, perm[:2]], device= self.device))        
        
        loss+= lambda_reg * ioss_penalty
        
        return loss
