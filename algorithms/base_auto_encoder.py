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
from models.decoder import Decoder

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helper import ValidationHelper
from utils.metrics import *

import wandb
wandb.init(project="polynomial-identification", reinit=True)

class AE():
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, seed=0, device= None):
        
        self.args= args
        self.seed= seed
        self.device= device
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
                
        self.encoder= Encoder(self.args.data_dim, self.args.latent_dim).to(self.device)
        self.decoder= Decoder(self.args.data_dim, self.args.latent_dim).to(self.device)
        
        self.opt, self.scheduler= self.get_optimizer()        
        self.validation_helper= ValidationHelper()
        
        if self.args.intervention_case:           
            self.res_dir= 'results/ae/intervention/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
        else:
            self.res_dir= 'results/ae/observation/' + str(self.args.save_dir) + 'seed_' + str(seed) + '/'
            
        self.save_path= self.res_dir        
        
        if self.args.wandb_log:
            wandb.init(project="polynomial-identification", reinit=True)
            wandb.run.name=  'ae/' + self.args.save_dir + 'seed_' + str(seed) + '/'
            
    def get_optimizer(self):        
                    
        opt= optim.Adam([
                    {'params': filter(lambda p: p.requires_grad, list(self.encoder.parameters()) + list(self.decoder.parameters()) )}, 
                    ], lr= self.args.lr, weight_decay= self.args.weight_decay )
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)        
        return opt, scheduler    
        
    def save_model(self):
        
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        
        torch.save(self.encoder.state_dict(), self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_encoder.pth')        
        
        torch.save(self.decoder.state_dict(), self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_decoder.pth')        
        return
    
    def load_model(self):
        
        self.encoder.load_state_dict(torch.load(self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_encoder.pth', map_location=torch.device('cpu')))
        self.encoder.eval()
        
        self.decoder.load_state_dict(torch.load(self.save_path + 'lr_' + str(self.args.lr) + '_weight_decay_' + str(self.args.weight_decay) + '_batch_size_' +  str(self.args.batch_size) + '_decoder.pth', map_location=torch.device('cpu')))
        self.decoder.eval()
        return        
    
    def validation(self):
        
        self.encoder.eval()  
        self.decoder.eval()
        
        val_loss=0.0
        count=0
        for batch_idx, (x, _, _) in enumerate(self.val_dataset):
            
            with torch.no_grad():
                                
                x= x.to(self.device)                
                z_pred= self.encoder(x)                    
                out= self.decoder(z_pred)
                loss= torch.mean(((out-x)**2)) 
                    
                val_loss+= loss.item()
                count+=1
            
            if self.args.wandb_log:
                wandb.log({'val_loss': val_loss/count})
        
        return val_loss/count
        
    def train(self):
        
        for epoch in range(self.args.num_epochs):            
                
            train_loss=0.0
            count=0
            
            #LR Scheduler
            self.scheduler.step()    
            print(self.scheduler.get_last_lr())
                                                    
            #Training            
            self.encoder.train()
            self.decoder.train()
            
            #Compute identification metrics
            if epoch % 10 == 0:
                self.eval_identification(epoch= epoch)
                if 'balls' in self.args.latent_case and ( epoch==0 or epoch==10 ):
                    self.save_intermediate_model(epoch= epoch)
            
            for batch_idx, (x, _, _) in enumerate(self.train_dataset):
                
                self.opt.zero_grad()
                
                #Forward Pass
                x= x.to(self.device)
                z_pred= self.encoder(x)
                x_pred= self.decoder(z_pred)
                
                #Compute Reconstruction Loss
                loss= self.compute_loss(z_pred, x_pred, x)
                    
                #Backward Pass
                loss.backward()
                
#                 grad_norm= 0.0            
#                 for p in self.coff_matrix.parameters():
#                     param_norm = p.grad.detach().data.norm(2)
#                     grad_norm += param_norm.item() ** 2
#                 grad_norm = grad_norm ** 0.5
#                 wandb.log({'grad_norm': grad_norm})
                
                self.opt.step()
                    
                train_loss+= loss.item()
                count+=1                
            
            val_score= self.validation()  
                                
            print('\n')
            print('Done Training for Epoch: ', epoch)
            print('Training Loss: ', train_loss/count)   
            print('Validation Loss: ', val_score)
            print('Best Epoch: ', self.validation_helper.best_epoch)   
            
            if self.args.wandb_log:
                wandb.log({'train_loss': train_loss/count})
                        
            if self.validation_helper.save_model(val_score, epoch):
                print('Saving model')
                self.save_model()
            elif self.validation_helper.early_stop(val_score):
                print('Early Stopping')
                break
            
        return
    
    def compute_loss(self, z_pred, x_pred, x):
        
        loss= torch.mean(((x-x_pred)**2))        
        return loss 
        
    
    def eval_identification(self, epoch= -1):
        
        self.encoder.eval()  
        self.decoder.eval()

        save_dir= 'results/plots/'+ self.args.save_dir + 'epoch_' + str(epoch) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        #Obtain Predictions and Reconstruction Loss
        if 'balls' in self.args.latent_case:        
            plot_case=True
        else:
            plot_case=False
        
        res= get_predictions(self.encoder, self.decoder, self.train_dataset, self.val_dataset, self.test_dataset, device= self.device, save_dir= save_dir, plot=plot_case)
        true_z= res['true_z']
        pred_z= res['pred_z']
        recon_err= res['recon_loss']
        
        print(true_z['tr'].shape, pred_z['tr'].shape)

        #Latent Prediction Error
        rmse, r2= get_indirect_prediction_error(pred_z, true_z)
        print('latent prediction r2: ', r2)        
        if 'balls' in self.args.latent_case:
            _, r2_mlp= get_indirect_prediction_error(pred_z, true_z, model= 'mlp')            
            print('latent prediction r2 MLP: ', r2_mlp)        
        else:
            r2_mlp= 0
        
        # MCC Score
        if 'balls' not in self.args.latent_case:
            mcc= get_cross_correlation(pred_z, true_z)
            print('MCC: ', mcc)
        else:
            mcc=0
        
        if self.args.wandb_log:
            wandb.log({'test_loss': recon_err['te']})
            wandb.log({'latent_pred_rmse': rmse})
            wandb.log({'latent_pred_r2': r2})
#             wandb.log({'max/min singular values': np.max(sig_values)/np.min(sig_values)})
            wandb.log({'mcc': mcc})
            wandb.log({'latent_pred_r2_mlp': r2_mlp})

#         #DCI
#         imp_matrix= compute_importance_matrix(pred_z['te'], true_z['te'], case='disentanglement')
#         score= disentanglement(imp_matrix)
#         wandb.log({'dci-disentanglement': score})        
        
#         imp_matrix= compute_importance_matrix(pred_z['te'], true_z['te'], case='completeness')
#         score= completeness(imp_matrix)
#         wandb.log({'dci-completeness': score})
        
#         #MCC
#         mcc= get_cross_correlation(pred_z, true_z)
#         wandb.log({'mcc': mcc})
        
        
        return rmse, r2

    
    def get_final_layer_weights(self):
        
        self.load_model()
        
        for p in self.model.fc_net.parameters():
            print(p.data)
            