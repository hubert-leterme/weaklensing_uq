#!/bin/bash
#SBATCH --mail-user hubert.leterme@cea.fr
#SBATCH --job-name=create_augmented_dataset
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --output ~/work/Log/slurm_out/out_%j.log

python scripts/create_augmented_dataset.py -z -b 10 --angle-batch-size 8 --angle-step 1 --niter-per-angle 2 -w 8 --seed 42 -v
