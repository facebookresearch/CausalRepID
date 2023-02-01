# Interventional Causal Representation Learning

## Reproduce Results for the Polynomial Decoder Datasets

### Generate Data 
- python scripts/main_exps.py --case data --target_latent uniform
- python scripts/main_exps.py --case data --target_latent uniform_corr
- python scripts/main_exps.py --case data --target_latent gaussian_mixture
- python scripts/main_exps.py --case data --target_latent scm_sparse
- python scripts/main_exps.py --case data --target_latent scm_dense

### Table 2: Observational Data Case

#### Train Models
- bash main_slurm_launcher 0 ae_poly 1e-3
- bash main_slurm_launcher 0 ae_poly 5e-4
- bash main_slurm_launcher 0 ae_poly 1e-4

#### Evaluate Models
- python scripts/main_exps.py --case test --method_type ae_poly --lr 1e-3 --intervention_case 0
- python scripts/main_exps.py --case test --method_type ae_poly --lr 5e-4 --intervention_case 0
- python scripts/main_exps.py --case test --method_type ae_poly --lr 1e-4 --intervention_case 0

#### Log Results 
- python scripts/main_exps.py --case log --method_type ae_poly --intervention_case 0


### Table 3: Interventional Data Case

Run all the commands stated above for the observational case (Table 2) with the flag `--intervention_case` set as 1

### Running experiments for Neural Network Decoder

Run all the commands stated above for the observational case (Table 2) with the flag `--method_type` set as 'ae'.


## Reproduce Results for the Image Dataset

### Table 4: Image Dataset

#### Train Models
- bash main_slurm_launcher_image.sh balls_uniform_none
- bash main_slurm_launcher_image.sh balls_scm_linear
- bash main_slurm_launcher_image.sh balls_scm_non_linear

#### Evaluate Models
- python scripts/main_exps_images.py --case test --target_latent balls_uniform_none
- python scripts/main_exps_images.py --case test --target_latent balls_scm_linear
- python scripts/main_exps_images.py --case test --target_latent balls_scm_non_linear

#### Log Results
- python scripts/main_exps_images.py --case log

## License

This source code is released under the MIT license, included [here](LICENSE).