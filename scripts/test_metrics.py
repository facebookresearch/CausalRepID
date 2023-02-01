import numpy as np
import scipy
import copy
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV

'''
z: True Latent (dataset_size, latent_dim) (.npy file)
Pred_z: Inferred Latent (dataset_size, latent_dim) (.npy file)
'''

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


# Test Cases

# Case 1: True latent same as the predicted latent

list_matrices= [ np.eye(10), np.random.random((10, 10)) ]

for matrix in list_matrices:

    true_z= np.random.random((1000, 10))
    pred_z= np.matmul(true_z, matrix)

    imp_matrix= compute_importance_matrix(pred_z, true_z, case='disentanglement')
    score= disentanglement(imp_matrix)
    print('Disentanglement', score)

    imp_matrix= compute_importance_matrix(pred_z, true_z, case='completeness')
    score= completeness(imp_matrix)
    print('Completeness', score)


# Permutation Case

pred_z= copy.deepcopy(true_z)
perm= np.random.permutation(10)
pred_z= pred_z[:, perm]

imp_matrix= compute_importance_matrix(pred_z, true_z, case='disentanglement')
score= disentanglement(imp_matrix)
print('Disentanglement', score)

imp_matrix= compute_importance_matrix(pred_z, true_z, case='completeness')
score= completeness(imp_matrix)
print('Completeness', score)
