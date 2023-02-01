#!/bin/sh
#SBATCH --cpus-per-task=4               
#SBATCH --mem=24G                        
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

#Force slurm output
export PYTHONUNBUFFERED=1

#Load module
module load python/3.8

#Load python environment
source ~/clear_rep_env/bin/activate

python train.py --latent_case $1 --method_type ae_image --lr 5e-4 --batch_size 64 --intervention_case 0 --cuda_device 0 --seed $2 --latent_dim 25 --num_epochs 1000 --wandb_log 0