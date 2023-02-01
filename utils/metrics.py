# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import copy
import torch
import torchvision
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from scipy.optimize import linear_sum_assignment
from sklearn.feature_selection import mutual_info_regression
import scipy

import matplotlib.pyplot as plt
from torchvision import transforms

def get_pca_sources(pred_z, pca_transform):
    
    return { 'tr': pca_transform.transform(pred_z['tr']), 'te': pca_transform.transform(pred_z['te']) }    

def get_ica_sources(pred_z, ica_transform):
    
    return { 'tr': ica_transform.transform(pred_z['tr']), 'te': ica_transform.transform(pred_z['te']) }

def regression_approx(x, y, model, fit_intercept=False):
    if model == 'lr':
        reg= LinearRegression(fit_intercept= fit_intercept).fit(x, y)
    elif model == 'lasso':
#         reg= Lasso(alpha=0.001, fit_intercept= True).fit(x, y)
        reg= LassoCV(fit_intercept=True, cv=3).fit(x, y)
    elif model == 'ridge':
#         reg= Ridge(alpha=0.001, fit_intercept= True).fit(x, y)
        alphas_list = np.linspace(1e-2, 1e0, num=10).tolist()
        alphas_list += np.linspace(1e0, 1e1, num=10).tolist()
        reg= RidgeCV(fit_intercept= True, cv=3, alphas=alphas_list).fit(x, y)
    elif model == 'mlp':
        reg= MLPRegressor(random_state=1, max_iter= 1000).fit(x, y)
        
    return reg

def get_predictions_check(train_dataset, test_dataset):    
    
    true_x={'tr':[], 'te':[]}
    true_z= {'tr':[], 'te':[]}
    
    data_case_list= ['train', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
    
        for batch_idx, (x, z, _) in enumerate(dataset):

            with torch.no_grad():
                                
                true_x[key].append(x)
                true_z[key].append(z)

        true_x[key]= torch.cat(true_x[key]).detach().numpy()
        true_z[key]= torch.cat(true_z[key]).detach().numpy()
    
    return true_x, true_z


def get_predictions(encoder, decoder, train_dataset, val_dataset, test_dataset, device= None, save_dir='plots/',  plot=True):
        
    true_z= {'tr':[], 'val':[], 'te':[]}
    pred_z= {'tr':[], 'val':[], 'te':[]}
    true_y= {'tr':[], 'val':[], 'te':[]}
    recon_loss= {'tr':0.0, 'val':0.0, 'te':0.0}
    
    data_case_list= ['train', 'val', 'test']
    for data_case in data_case_list:
        
        if data_case == 'train':
            dataset= train_dataset
            key='tr'
        elif data_case == 'val':
            dataset= val_dataset
            key='val'
        elif data_case == 'test':
            dataset= test_dataset
            key='te'
        
        count=0
        for batch_idx, (x, z, y) in enumerate(dataset):
            with torch.no_grad():
                
                x= x.to(device)
                pred= encoder(x)
                x_pred= decoder(pred)
                loss= torch.mean((x-x_pred)**2)
                
                true_y[key].append(y)
                true_z[key].append(z)
                pred_z[key].append(pred)
                recon_loss[key]+= loss.item()
                count+=1                
                
                if plot and batch_idx == 0:
                    for idx in range(5):
                        
                        transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                        ]
                        )                        
                        
                        data= x[idx].cpu()
                        data= (data - data.min()) / (data.max() - data.min())
                        data= transform(data)                        
                        data.save( save_dir  + 'real_image_' + str(idx) + '.jpg')

                        data= x_pred[idx].cpu()
                        data= (data - data.min()) / (data.max() - data.min())
                        data= transform(data)
                        data.save( save_dir  + 'fake_image_' + str(idx) + '.jpg')


        true_y[key]= torch.cat(true_y[key]).detach().numpy()
        true_z[key]= torch.cat(true_z[key]).detach().numpy()
        pred_z[key]= torch.cat(pred_z[key]).cpu().detach().numpy()
        recon_loss[key]= recon_loss[key]/count
    
#     print('Sanity Check: ')
#     print( true_y['tr'].shape, pred_y['tr'].shape, true_z['tr'].shape, pred_z['tr'].shape )
#     print( true_y['te'].shape, pred_y['te'].shape, true_z['te'].shape, pred_z['te'].shape )
    return {'true_z': true_z, 'pred_z': pred_z, 'true_y': true_y, 'recon_loss': recon_loss}


def get_indirect_prediction_error(pred_latent, true_score, case='test', model='lr'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
                
    reg= regression_approx(pred_latent['tr'], true_score['tr'], model, fit_intercept=True)
    pred_score= reg.predict(pred_latent[key])
    if len(pred_score.shape) == 1:
        pred_score= np.reshape(pred_score, (pred_score.shape[0], 1))
    
    rmse= np.sqrt(np.mean((true_score[key] - pred_score)**2))
    r2= r2_score(true_score[key], pred_score)  
        
#     mat= reg.coef_
#     _, sig_values ,_ = np.linalg.svd(mat)
#     print(mat)
#     print(np.mean(pred_latent['tr']), np.var(pred_latent['tr']))
#     sys.exit()
            
    return rmse, r2

    
def get_mi_score(pred_latent, true_latent, case='test'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    n= pred_latent[key].shape[0]
    dim= pred_latent[key].shape[1]
    mutual_info= 0.0
    for i in range(dim):
        for j in range(dim):
            if i != j:
                mutual_info+= mutual_info_regression( np.reshape( pred_latent[key][:, i], (n, 1) ), true_latent[key][:, j] )
    
    print('Mutual Information')
    print(mutual_info/(dim**2 - dim))
    return 

    
def get_independence_score(pred_latent, true_latent, case='test'):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    dim= pred_latent[key].shape[1]
    cross_corr= np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (np.cov( pred_latent[key][:,i], true_latent[key][:,j] )[0,1]) / ( np.std(pred_latent[key][:,i])*np.std(true_latent[key][:,j]) )
    
    print('Independence Score')
    print(cross_corr)
    print(np.linalg.norm( cross_corr - np.eye(dim),  ord='fro'))
    return 

def get_cross_correlation(pred_latent, true_latent, case='test', batch_size= 5000):
    
    if case == 'train':
        key= 'tr'
    elif case == 'test':
        key= 'te'
    
    num_samples= pred_latent[key].shape[0]
    dim= pred_latent[key].shape[1]
    total_batches= int( num_samples / batch_size )  

    mcc_arr= []
    for batch_idx in range(total_batches):
        
        z_hat= copy.deepcopy( pred_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        z= copy.deepcopy( true_latent[key][ (batch_idx)*batch_size : (batch_idx+1)*batch_size ] )
        batch_idx += 1
        
        cross_corr= np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                cross_corr[i,j]= (np.cov( z_hat[:,i], z[:,j] )[0,1]) / ( np.std(z_hat[:,i])*np.std(z[:,j]) )

#         cross_corr= np.corrcoef(pred_latent[key], true_latent[key], rowvar=False)[dim:, :dim]
        cost= -1*np.abs(cross_corr)
        row_ind, col_ind= linear_sum_assignment(cost)
        score= 100*( -1*cost[row_ind, col_ind].sum() )/(dim)
        print(-100*cost[row_ind, col_ind])
    #     score= 100*np.sum( -1*cost[row_ind, col_ind] > 0.80 )/(dim)
    
        mcc_arr.append(score)
    
    return mcc_arr


def intervene_metric(pred_latent, true_latent, model='lr', model_train=1, list_models=None, hard_intervention_val= 2.0):
    
    '''
    pred_latent: Output representation from stage 1 
    true_score: intervened latent true value
    '''
    latent_dim= true_latent['tr'].shape[1]
    
    if model_train:
        res={}
        for intervene_idx in range(latent_dim):
            
            indices= true_latent['tr'][:, intervene_idx] == hard_intervention_val
            curr_pred_latent_subset=  pred_latent['tr'][indices]
            intervene_targets= 10* np.ones( curr_pred_latent_subset.shape[0] )

            reg= regression_approx(curr_pred_latent_subset, intervene_targets, model, fit_intercept=False)
            res[intervene_idx]= reg
            
        return res
            
    else:
        num_samples= true_latent['te'].shape[0]
        res= np.zeros((num_samples, latent_dim))
        
        for intervene_idx in range(latent_dim):
            res[:, intervene_idx]= list_models[intervene_idx].predict(pred_latent['te'])

        return {'te': res}
    
    
def intervene_metric_image(pred_latent, true_latent, intervention_meta_data, model='lr', model_train=1, list_models=None):
    
    '''
    pred_latent: Output representation from stage 1 
    true_score: intervened latent true value
    '''
    
    if model_train:
                
        res={}        
        
        intervened_latents_set= np.unique( intervention_meta_data['tr'][:, 0] )
        print(intervened_latents_set)
        for intervene_idx in intervened_latents_set:
            print(intervene_idx)
            
            indices= intervention_meta_data['tr'][:, 0] ==  intervene_idx
            curr_pred_latent_subset=  pred_latent['tr'][indices]
            
            intervene_targets= 10* intervention_meta_data['tr'][indices, 1]
            print(np.unique(intervene_targets))
        
            reg= regression_approx(curr_pred_latent_subset, intervene_targets, model, fit_intercept=False)
            res[intervene_idx]= reg        
            
        return res
            
    else:
        
        intervened_latents_set= list(list_models.keys())
        
        num_samples= true_latent['te'].shape[0]        
        eff_latent_dim= len(intervened_latents_set)
        
        z= np.zeros((num_samples, eff_latent_dim))
        z_hat= np.zeros((num_samples, eff_latent_dim))
        
        for idx in range(eff_latent_dim):
            intervene_idx= intervened_latents_set[idx]
            
            z[:, idx]= true_latent['te'][:, int(intervene_idx)]
            z_hat[:, idx]= list_models[intervene_idx].predict(pred_latent['te'])
        
        print('Transformed Latents using Itv Dataset', z_hat.shape, z.shape)
        return {'te':z_hat}, {'te': z}
    
    
# DCI Score
def compute_importance_matrix(z_pred, z, case= 'disentanglement', fit_intercept= True):
    
    true_latent_dim= z.shape[1]
    pred_latent_dim= z_pred.shape[1]
    imp_matrix= np.zeros((pred_latent_dim, true_latent_dim))
    for idx in range(true_latent_dim):
        model= LinearRegression(fit_intercept= fit_intercept).fit(z_pred, z[:, idx])
#         model= LassoCV(fit_intercept=True, cv=3).fit(z_pred, z[:, idx])
        imp_matrix[:, idx]= model.coef_
    
    # Taking the absolute value for weights to encode relative importance properly
    imp_matrix= np.abs(imp_matrix)
    if case == 'disetanglement':
        imp_matrix= imp_matrix / np.reshape( np.sum(imp_matrix, axis=1), (pred_latent_dim, 1) )
    elif case == 'completeness':
        imp_matrix= imp_matrix / np.reshape( np.sum(imp_matrix, axis=0), (1, true_latent_dim) )
    
    return  imp_matrix
    

def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])

def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])

def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)
