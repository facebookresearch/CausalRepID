# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='log',
                   help= 'test; log; debug')
parser.add_argument('--target_latent', type=str, default='balls_iid_none',
                   help= '')
parser.add_argument('--eval_ioss_transformation', type=int, default=0,
                   help='Evaluate the IOSS transformation from the base model representation')
parser.add_argument('--eval_intervene_transformation', type=int, default=1,
                   help='Evaluate the Intervention transformation from the base model representation')
args = parser.parse_args()

case= args.case
target_latent= args.target_latent
eval_ioss_transformation= args.eval_ioss_transformation
eval_intervene_transformation= args.eval_intervene_transformation

latent_case_grid= ['balls_iid_none', 'balls_scm_linear', 'balls_scm_non_linear']
interventions_per_latent_grid= [1, 3, 5, 7, 9]
method_type= 'ae_image'
total_seeds= 5
latent_dim =  25
lr= 5e-4
intervention_case= 0
batch_size= 64
cuda_device= 0
            
#Test Models
if args.case == 'test':
    for latent_case in latent_case_grid:
        if latent_case != target_latent:
            continue
        
        for total_interventions in interventions_per_latent_grid:
            
            latent= latent_case.split('_')[1]
            mechanism= latent_case.split('_')[2]
            if mechanism == 'non':
                mechanism= 'non_linear'
            
            #Data Script
            script= 'python data/balls_dataset.py --distribution_case intervention ' + ' --latent_case ' + str(latent) +  ' --scm_mechanism ' + str(mechanism) + ' --interventions_per_latent ' + str(total_interventions)
            print('Data Case: ', script)
            os.system(script)            
        
            #Eval Script
            fname= 'latent_case_' + str(latent_case) + '_itv_per_latent_' + str(total_interventions)  +  '_latent_dim_' + str(latent_dim) + '_lr_' + str(lr) + '_method_type_' + str(method_type) + '.txt'
            print(fname)

            script= 'python test.py ' +  ' --latent_case ' + str(latent_case) +  ' --latent_dim ' + str(latent_dim) + ' --intervention_case ' + str(intervention_case) + ' --lr ' + str(lr) + ' --batch_size ' + str(batch_size) + ' --method_type ' + str(method_type) + ' --num_seeds ' + str(total_seeds)  + ' --cuda_device ' + str(cuda_device) + ' --eval_ioss_transformation ' + str(eval_ioss_transformation) + ' --eval_intervene_transformation ' + str(eval_intervene_transformation)  + ' > ' + 'results/final_logs/balls/' + fname
            os.system(script)

                    
#Log Results

latent_name_map= {'balls_iid_none': 'Uniform', 'balls_scm_linear': 'SCM (linear)', 'balls_scm_non_linear': 'SCM (non-linear)'}

if args.case == 'log':
    
    meta_res={}
    for lr in [0.0005]:
        res={'Latent Case': [], 'Total Interventions': [], 'Recon-RMSE':[], 'R2':[], 'MCC-Tune': []}
        
        for latent_case in latent_case_grid:        
            for total_interventions in interventions_per_latent_grid:

                fname= 'latent_case_' + str(latent_case) + '_itv_per_latent_' + str(total_interventions)  +  '_latent_dim_' + str(latent_dim) + '_lr_' + str(lr) + '_method_type_' + str(method_type) + '.txt'
                data= open( 'results/final_logs/balls/' + fname, 'r').readlines()

                for line in data:
                    line= line.replace('\n','')
                    if 'Metric' in line:

                        if 'recon_rmse' in line:
                            key= 'Recon-RMSE'
                        elif 'latent_pred_r2' in line:
                            key= 'R2'
                        elif 'mcc_tune' in line:
                            key= 'MCC-Tune'
                        else:
                            continue

                        mean= round( float(line.split(" ")[-2]), 2 )
                        var= round( float(line.split(" ")[-1]), 2 )

                        val=  str(mean) + ' \u00B1 ' + str(var)
#                         val=  str(mean) + ' ( ' + str(var)  + ' ) '
                        res[key].append(val)
                
                res['Latent Case'].append( latent_name_map[latent_case] )
                res['Total Interventions'].append(total_interventions)
                    
        meta_res[lr]= res
    
    final_res={'Latent Case': [], 'Total Interventions': [], 'LR': [], 'Recon-RMSE':[], 'MCC-Tune': []}
    
    final_res= meta_res[0.0005]

    for key in final_res.keys():
        print(key, len(final_res[key]))
    df= pd.DataFrame(final_res)
    
#     df= df.drop(columns= ['Recon-RMSE'])
    print(df.to_latex(index=False))        
