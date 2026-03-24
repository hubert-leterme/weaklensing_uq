#!/bin/bash
#SBATCH --job-name=create_calib_dataset_brightfaint_zbins
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
source /opt/ohpc/pub/apps/anaconda/3.9/2021.11/etc/profile.d/conda.sh
conda activate wlmmuq

cd /feynman/home/dap/lcs/hl285110/Documents/Code/weaklensing

srun python -u scripts/create_augmented_dataset.py -o $HOME/work/Data/kappaTNG_processed/calib/LP001_augmented_brightfaint_zbins_384.hdf5 --idx-lp 1 -z --angle-batch-size 5 --angle-step 8 --cosmos-include-faint -w 5 --seed 42 -v
