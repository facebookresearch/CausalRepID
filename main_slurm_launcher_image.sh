latent_case=$1

for seed in 0 1 2 3 4
do
    sbatch scripts/slurm_exps_images.sh $latent_case $seed
done

