# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/sh
#SBATCH --cpus-per-task=4               
#SBATCH --mem=24G                        
#SBATCH --time=12:00:00

#Force slurm output
export PYTHONUNBUFFERED=1

#Load module
module load python/3.8

#Load python environment
source ~/clear_rep_env/bin/activate

python3 scripts/main_exps.py --case $1 --intervention_case $2 --lr $3 --method_type $4 --target_latent $5 --target_dim $6 --target_degree $7