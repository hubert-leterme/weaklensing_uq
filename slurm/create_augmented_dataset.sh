#!/bin/bash
#SBATCH --mail-user hubert.leterme@cea.fr
#SBATCH --job-name create_augmented_dataset
#SBATCH --time 04:00:00
#SBATCH --cpus-per-task 4
#SBATCH --output slurm/out/output.log
#SBATCH --error slurm/out/error.log

python scripts/create_augmented_dataset.py --angle-step 1 --niter-per-angle 2 --seed 42 -v
