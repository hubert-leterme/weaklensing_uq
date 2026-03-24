#!/bin/bash
#SBATCH --job-name=create_trainval_dataset_brightfaint
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
source /opt/ohpc/pub/apps/anaconda/3.9/2021.11/etc/profile.d/conda.sh
conda activate wlmmuq

cd /feynman/home/dap/lcs/hl285110/Documents/Code/weaklensing

srun python -u scripts/create_augmented_dataset.py -o $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_brightfaint_384.hdf5 --idx-lp 2 --angle-batch-size 8 --angle-step 1 --niter-per-angle 2 --cosmos-include-faint -w 8 --seed 42 -v
