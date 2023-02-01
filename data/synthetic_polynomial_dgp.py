# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#Common imports
import sys
import os
import argparse
import random
import copy
import math

import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.stats import bernoulli

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from data.dag_generator import DagGenerator

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_total_polynomial_terms(poly_degree, latent_dim):
    count=0
    for degree in range(poly_degree+1):
        count+= pow(latent_dim, degree)
    return count

def compute_kronecker_product(degree, latent):
    if degree ==0:
        out= np.array([1])        
    else:
        out=copy.deepcopy(latent)
        for idx in range(1, degree):
            out= np.kron(out, latent)
#     print(out.shape)    
    return out

def compute_decoder_polynomial(poly_degree, latent):
    out=[]
    for degree in range(poly_degree+1):
#         print('Computing polynomial term of degree ', degree)
        out.append(compute_kronecker_product(degree, latent))
        
    out= np.concatenate(out)
    out= np.reshape(out, (1,out.shape[0]))    
    return out


def generate_latent_vector(dataset_size, latent_dim, latent_case, intervention_indices= [], intervention_case= 0, dag=None, base_dir= ''):

    z= np.zeros((dataset_size, latent_dim))
    
    for i in range(latent_dim):
        if latent_case == 'laplace':
            z[:, i]= np.random.laplace(10, 5, dataset_size)
        elif latent_case == 'uniform':
            z[:, i]= np.random.uniform(low=-5, high=5, size=dataset_size)
        elif latent_case == 'uniform_discrete':
            z[:, i]= np.random.randint(10, size= dataset_size)                
        elif latent_case == 'gaussian':
            z[:, i]= np.random.normal(0, 1, size= dataset_size)
        elif latent_case == 'special':
            support= np.array([0, 1])
            p= 1
            prob= np.exp( -1*support**p )
            prob= prob/np.sum(prob)
            idx= np.argmax( np.random.multinomial(1, prob, size=dataset_size), axis=1 )
            z[:, i]= support[idx]                
                        
    if latent_case == 'gaussian_corr':
        rho= 0.9
        noise_var= np.eye(latent_dim)
        for i in range(latent_dim):
            for j in range(i+1, latent_dim):
                noise_var[i,j]=  rho ** (np.abs(i-j))
                
        z= np.random.multivariate_normal(np.zeros(latent_dim), noise_var, dataset_size)
        
    if latent_case == 'uniform_corr':
        for d_idx in range(0 , latent_dim, 2):
            print('Latent entries for the pair: ', d_idx, d_idx + 1)
            p1= bernoulli.rvs(0.5, size=dataset_size)
            p2= bernoulli.rvs(0.9, size=dataset_size)
            z_11= np.random.uniform(low=0, high=5, size=dataset_size)
            z_12= np.random.uniform(low=-5, high=0, size=dataset_size)
            z_21= np.random.uniform(low=0, high=3, size=dataset_size)
            z_22= np.random.uniform(low=-3, high=0, size=dataset_size)

            for idx in range(dataset_size):
                if p1[idx] == 1:
                    z[idx, d_idx + 0]= z_11[idx]
                    if p2[idx] == 1:
                        z[idx, d_idx + 1]= z_21[idx]
                    else:
                        z[idx, d_idx + 1]= z_22[idx]
                else:
                    z[idx, d_idx + 0]= z_12[idx]
                    if p2[idx] == 1:
                        z[idx, d_idx + 1]= z_22[idx]
                    else:
                        z[idx, d_idx + 1]= z_21[idx]
    
    if 'mixture' in latent_case:
        mix_coff= bernoulli.rvs(0.5, size=dataset_size)
        mix_coff= np.reshape( mix_coff, (mix_coff.shape[0], 1) )
        
        z1= np.zeros((dataset_size, latent_dim))
        z2= np.zeros((dataset_size, latent_dim))
        for i in range(latent_dim):
            if args.latent_case == 'uniform_mixture':
                z1[:, i]= np.random.uniform(low=-5, high=5, size=dataset_size)
                z2[:, i]= np.random.uniform(low=-1, high=1, size=dataset_size)
            elif args.latent_case == 'gaussian_mixture':
                z1[:, i]= np.random.normal(0, 1, size=dataset_size)
                z2[:, i]= np.random.normal(1, 2, size= dataset_size)
            else:
                print('Error: Not valid latent type for a mixture model')
                sys.exit()
        
        z= mix_coff * z1 + (1-mix_coff) * z2
            
    if intervention_case and 'scm' not in latent_case:
        for idx, intervene_idx in np.ndenumerate(intervention_indices):
            z[idx, intervene_idx]= 2.0
    
    if 'scm' in latent_case:        
        if intervention_case:
            latent_dict = {}
            for intervene_idx in range(latent_dim):
                dag_copy= copy.deepcopy(dag)
                df, obj= dag_copy.intervene(intervention_nodes= [intervene_idx], target_distribution= 'hard_intervention')
                latent_dict[intervene_idx]= df
            for idx, intervene_idx in np.ndenumerate(intervention_indices):
                z[idx, :]= latent_dict[intervene_idx][idx, :]            
        else:            
            df, obj= dag.generate()
            z= df.values[:dataset_size, :]
        
        nx.draw_networkx(obj, arrows=True)
        plt.savefig( base_dir + latent_case  + '.jpg')
        plt.clf()
    
    return z
    
'''
Notations:
n: batch size
d: latent dimension
D: data dimension
p: degree of polynomial
q: number of terms in polynomial


Dimensions:
p: 2; q~ 111
z: (n, d): d= 10
x: (n, D): D= 100 -> 25
Poly(z): (n, q)
Coefficient Matrix: (D, q)
Ideal: D > q
'''

#TODO: Throw exception when the latent case if not amongst the valid ones
    
# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--data_dim', type=int, default=200,
                    help='')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='')
parser.add_argument('--latent_case', type=str, default='uniform',
                    help='uniform; uniform_corr; gaussian_mixture')
parser.add_argument('--poly_degree', type=int, default=2,
                    help='')
parser.add_argument('--train_size', type=int, default=10000,
                    help='')
parser.add_argument('--test_size', type=int, default=20000,
                    help='')

args = parser.parse_args()
seed= args.seed
data_dim= args.data_dim
latent_dim= args.latent_dim
latent_case= args.latent_case
poly_degree= args.poly_degree
train_size= args.train_size
test_size= args.test_size

poly_size= compute_total_polynomial_terms(poly_degree, latent_dim)
print('Total Polynomial Terms: ', poly_size)
    
#Random Seed
random.seed(seed*10)
np.random.seed(seed*10) 

coff_matrix= np.random.multivariate_normal(np.zeros(poly_size), np.eye(poly_size), size=data_dim).T
print('Coeff Matrix', coff_matrix.shape)
_, sng_values, _ = np.linalg.svd(coff_matrix)
# print('Singular Values for Coeff Matrix: ', sng_values)

#DAG
#NOTE: For this configuration to work we need to have the same train and test size
if latent_case == 'scm_sparse':
    dag= DagGenerator('linear', cause='gaussian', nodes=latent_dim, npoints= max( args.train_size, args.test_size), expected_density= 0.5)
elif latent_case == 'scm_dense':
    dag= DagGenerator('linear', cause='gaussian', nodes=latent_dim, npoints= max( args.train_size, args.test_size), expected_density= 1.0)
else:
    dag= None
    
for distribution_case in ['observational', 'interventional']:
    for data_case in ['train', 'val', 'test']: 
        
        if distribution_case == 'observational':
            base_dir= 'data/datasets/' + 'seed_' + str(seed) + '/observation/'
        elif distribution_case == 'interventional':
            base_dir= 'data/datasets/' + 'seed_' + str(seed) + '/intervention/'

        base_dir= base_dir + 'polynomial_latent_' + latent_case + '_poly_degree_' + str(poly_degree) + '_data_dim_' + str(data_dim) + '_latent_dim_' + str(latent_dim)  + '/' 
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)        

        print('Data Case: ', data_case)
        if data_case == 'train':
            dataset_size= args.train_size
        if data_case == 'val':
            dataset_size= int(args.train_size/4)
        elif data_case == 'test':
            dataset_size= args.test_size
            
        #Generating the latent vector
        if distribution_case == 'observational':
            y= -1 * np.ones(dataset_size)
            z= generate_latent_vector(dataset_size, latent_dim, latent_case, intervention_case= 0, intervention_indices= y, dag= dag, base_dir= base_dir)
        elif distribution_case == 'interventional':
            y= np.argmax(np.random.multinomial(1, [1/latent_dim]*latent_dim, dataset_size), axis=1 )
            z= generate_latent_vector(dataset_size, latent_dim, latent_case, intervention_case= 1, intervention_indices= y, dag= dag, base_dir= base_dir)
        
        print('Latent Z')    
        print(z.shape)
        print(z[:5])

        #Transforming the latent via polynomial decoder
        print('Data X')
        x=[]
        for idx in range(z.shape[0]):
            x.append( compute_decoder_polynomial(poly_degree, z[idx, :] ) )
        x= np.concatenate(x, axis=0)
        print(x.shape)

        x1= np.matmul(x[:, :1+latent_dim], coff_matrix[:1+latent_dim, :])
        print('X1')
        print('Min', np.min(np.abs(x1)), 'Max', np.max(np.abs(x1)), 'Mean', np.mean(np.abs(x1)))

        x2= np.matmul(x[:, 1+latent_dim:], coff_matrix[1+latent_dim:, :])
        norm_factor= 0.5 * np.max(np.abs(x2)) / np.max(np.abs(x1)) 
        x2 = x2 / norm_factor
        print('X2')
        print('Min', np.min(np.abs(x2)), 'Max', np.max(np.abs(x2)), 'Mean', np.mean(np.abs(x2)))

        x= (x1+x2)
        print(x.shape)

        f= base_dir + data_case + '_' + 'x' + '.npy'
        np.save(f, x)

        f= base_dir + data_case + '_' + 'z' + '.npy'
        np.save(f, z)

        f= base_dir + data_case + '_' + 'y' + '.npy'
        np.save(f, y)
