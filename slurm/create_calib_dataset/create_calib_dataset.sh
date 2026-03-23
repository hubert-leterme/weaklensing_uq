#!/bin/bash
#SBATCH --job-name=create_calib_dataset
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

srun python -u scripts/create_augmented_dataset.py -o $HOME/work/Data/kappaTNG_augmented/calib_384.hdf5 --idx-lp 1 --angle-batch-size 5 --angle-step 8 -w 5 --seed 42 -v
