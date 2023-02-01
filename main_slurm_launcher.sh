# Example full command
# sbatch scripts/slurm_exps.sh train 0 1e-3 ae uniform 6 2

intervention_case=$1
method_type=$2
lr=$3

for latent_dim in 6 10
do
    for poly_degree in 2 3
    do
        sbatch scripts/slurm_exps.sh train $intervention_case $lr $method_type uniform $latent_dim $poly_degree
        sbatch scripts/slurm_exps.sh train $intervention_case $lr $method_type uniform_corr $latent_dim $poly_degree
        sbatch scripts/slurm_exps.sh train $intervention_case $lr $method_type gaussian_mixture $latent_dim $poly_degree
        sbatch scripts/slurm_exps.sh train $intervention_case $lr $method_type scm_sparse $latent_dim $poly_degree
        sbatch scripts/slurm_exps.sh train $intervention_case $lr $method_type scm_dense $latent_dim $poly_degree
    done
done
