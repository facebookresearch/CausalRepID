import os
import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='log',
                   help= 'train; test; log; debug')
parser.add_argument('--intervention_case', type=int, default=0,
                   help='')
parser.add_argument('--target_latent', type=str, default='uniform',
                   help= '')
parser.add_argument('--target_dim', type=int, default=6,
                   help='')
parser.add_argument('--target_degree', type=int, default=2,
                   help='')
parser.add_argument('--method_type', type=str, default='ae_poly',
                   help= 'ae, ae_poly')
parser.add_argument('--batch_size', type=int, default= 16,
                    help='')
parser.add_argument('--lr', type=float, default= 1e-3,
                    help='')
parser.add_argument('--train_base_model', type=int, default=1,
                   help='Train the base auto encoder')
parser.add_argument('--train_ioss_transformation', type=int, default=0,
                   help='Learn the IOSS transformation from the base model representations')
parser.add_argument('--eval_ioss_transformation', type=int, default=0,
                   help='Evaluate the IOSS transformation from the base model representation')
parser.add_argument('--eval_intervene_transformation', type=int, default=0,
                   help='Evaluate the Intervention transformation from the base model representation')
parser.add_argument('--eval_dgp', type=int, default= 1,
                    help= 'Evaluate the function from z -> x and x -> z in the true DGP')
parser.add_argument('--cuda_device', type=int, default=-1, 
                    help='Select the cuda device by id among the avaliable devices' )
args = parser.parse_args()

intervention_case= args.intervention_case
batch_size= args.batch_size
lr= args.lr
method_type= args.method_type
target_latent= args.target_latent
target_dim= args.target_dim
target_degree= args.target_degree
train_base_model= args.train_base_model
train_ioss_transformation= args.train_ioss_transformation
eval_ioss_transformation= args.eval_ioss_transformation
eval_intervene_transformation= args.eval_intervene_transformation
eval_dgp= args.eval_dgp
cuda_device= args.cuda_device

# latent_case_grid= ['uniform', 'uniform_corr']
# latent_case_grid= ['gaussian_mixture', 'scm_sparse', 'scm_dense']
latent_case_grid= ['uniform', 'uniform_corr', 'gaussian_mixture', 'scm_sparse', 'scm_dense']
# latent_case_grid= ['scm_sparse', 'scm_dense']
poly_degree_grid= [2, 3]
latent_dim_grid=[6, 10]
total_seeds= 5
data_dim = 200

#Sanity Checks
if args.case == 'debug':
    for latent_case in latent_case_grid:        
        for latent_dim in latent_dim_grid:
            for poly_degree in poly_degree_grid:
                for seed in range(total_seeds):
                    
                    curr_dir= 'results/ae-ioss/ae_poly/' + 'polynomial' + '_latent_' + latent_case + '_poly_degree_' + str(poly_degree) + '_data_dim_' + str(data_dim) + '_latent_dim_' + str(latent_dim)  + '/' + 'seed_' + str(seed) + '/' 
#                     curr_dir= 'results/ae-poly/' + 'polynomial' + '_latent_' + latent_case + '_poly_degree_' + str(poly_degree) + '_data_dim_' + str(data_dim) + '_latent_dim_' + str(latent_dim)  + '/' + 'seed_' + str(seed) + '/' 
                    
                    count=0
                    for _, _, f_list in os.walk(curr_dir):
                        for fname in f_list:
                            if '.pth' in fname:
                                count+=1

                    if count!=6:
                        print('Error: ', latent_case, latent_dim, poly_degree, seed, count)
                    
#Generate Datasets
if args.case == 'data':    
    for latent_case in latent_case_grid:
        
        if latent_case != target_latent:
            continue
        
        for latent_dim in latent_dim_grid:
            for poly_degree in poly_degree_grid:
                for seed in range(total_seeds):
                    script= 'python data/synthetic_polynomial_dgp.py ' +  ' --latent_case ' + str(latent_case) +  ' --latent_dim ' + str(latent_dim) + ' --poly_degree ' + str(poly_degree) + ' --seed ' + str(seed)
                    os.system(script)
            
#Train Models
if args.case == 'train':
    
    train_ioss_transformation= 1 - intervention_case
    
    for latent_case in latent_case_grid:
        
        if latent_case != target_latent:
            continue
        
        for latent_dim in latent_dim_grid:
            
            if latent_dim != target_dim:
                continue
            
            for poly_degree in poly_degree_grid:
                
                if poly_degree != target_degree:
                    continue
                
                for seed in range(total_seeds):
                    script= 'python train.py ' +  ' --latent_case ' + str(latent_case) +  ' --latent_dim ' + str(latent_dim) + ' --poly_degree ' + str(poly_degree) + ' --seed ' + str(seed) + ' --intervention_case ' + str(intervention_case) + ' --lr ' + str(lr) + ' --batch_size ' + str(batch_size) + ' --method_type ' + str(method_type) + ' --cuda_device ' + str(cuda_device) + ' --train_base_model ' + str(train_base_model) + ' --train_ioss_transformation ' + str(train_ioss_transformation)
                    os.system(script)
                
#Test Models
if args.case == 'test':
    
    eval_ioss_transformation= 1 - intervention_case
    eval_intervene_transformation= intervention_case
    
    for latent_case in latent_case_grid:        
        for latent_dim in latent_dim_grid:
            for poly_degree in poly_degree_grid:
                fname= 'latent_case_' + str(latent_case) +  '_latent_dim_' + str(latent_dim) + '_poly_degree_' + str(poly_degree) + '_intervention_' + str(intervention_case) + '_lr_' + str(lr) + '_method_type_' + str(method_type) + '.txt'
                print(fname)
                script= 'python test.py ' +  ' --latent_case ' + str(latent_case) +  ' --latent_dim ' + str(latent_dim) + ' --poly_degree ' + str(poly_degree) + ' --intervention_case ' + str(intervention_case) + ' --lr ' + str(lr) + ' --batch_size ' + str(batch_size) + ' --method_type ' + str(method_type)  + ' --eval_ioss_transformation ' + str(eval_ioss_transformation) + ' --eval_intervene_transformation ' + str(eval_intervene_transformation)  +  ' --eval_dgp ' + str(eval_dgp) + ' > ' + 'results/final_logs/'+ fname
                os.system(script)            

            

#Log Results

latent_name_map= {'uniform': 'Uniform', 'uniform_corr': 'Uniform-C', 'gaussian_mixture': 'Gaussian-Mixture', 'scm_sparse': 'SCM-S', 'scm_dense': 'SCM-D'}

if args.case == 'log':
    
    meta_res={}
    for lr in [0.001, 0.0005, 0.0001]:
        res={'Latent Case': [], 'Latent Dim': [], 'Poly Degree': [], 'Recon Error':[], 'RMSE': [], 'R2': [], 'Debug-R2' : [], 'Oracle-R2': [], 'MCC': [], 'MCC-Tune': []}
        
        for latent_case in latent_case_grid:        
            for latent_dim in latent_dim_grid:
                for poly_degree in poly_degree_grid:

                    fname= 'latent_case_' + str(latent_case) +  '_latent_dim_' + str(latent_dim) + '_poly_degree_' + str(poly_degree) + '_intervention_' + str(intervention_case) + '_lr_' + str(lr) + '_method_type_' + str(method_type) + '.txt'
                    data= open( 'results/final_logs/' + fname, 'r').readlines()

                    for line in data:
                        line= line.replace('\n','')
                        if 'Metric' in line:

                            if 'recon_rmse' in line:                                
                                key= 'Recon Error'                                                             
                            elif 'latent_pred_rmse' in line:
                                key= 'RMSE'
                            elif 'latent_pred_r2' in line:
                                key= 'R2'
                            elif 'mcc ' in line:
                                key= 'MCC'
                            elif 'mcc_tune' in line:
                                key= 'MCC-Tune'
                            elif 'debug_pred_r2' in line:
                                key= 'Debug-R2'
                            elif 'oracle_pred_r2' in line:
                                key= 'Oracle-R2'
                            else:
                                continue                            
                                
                            mean= round( float(line.split(" ")[-2]), 2 )
                            var= round( float(line.split(" ")[-1]), 2 )
                            
#                             val=  str(mean) + ' ( ' + str(var)  + ' ) '
                            val=  str(mean) +  ' \u00B1 ' + str(var)
                            res[key].append(val)

                    res['Latent Case'].append( latent_name_map[latent_case] )
                    res['Latent Dim'].append(latent_dim)
                    res['Poly Degree'].append(poly_degree)
        meta_res[lr]= res
    
    
    final_res={'Latent Case': [], 'Latent Dim': [], 'Poly Degree': [], 'LR': [], 'Recon Error':[], 'RMSE': [], 'R2': [], 'Debug-R2' : [], 'Oracle-R2': [], 'MCC': [], 'MCC-Tune': []}
    
    
    total_size= len(res['Recon Error'])    
    for idx in range(total_size):
        
        opt_val= np.inf
        opt_lr= -1
        for lr in [0.001, 0.0005, 0.0001]:
            curr_val= float(meta_res[lr]['Recon Error'][idx].split(' ')[0])
            if opt_val > curr_val:
                opt_val = curr_val
                opt_lr= lr
        final_res['LR'].append(opt_lr)        
        for key in res.keys():
            final_res[key].append( meta_res[opt_lr][key][idx] )

    for key in final_res.keys():
        print(key, len(final_res[key]))
    df= pd.DataFrame(final_res)
    
    df= df.drop(columns= ['RMSE', 'Debug-R2', 'Oracle-R2', 'LR'])
#     df= df.drop(columns= ['RMSE', 'Debug-R2', 'Oracle-R2', 'LR', 'R2', 'Recon Error'])

    print(df.to_latex(index=False))        
