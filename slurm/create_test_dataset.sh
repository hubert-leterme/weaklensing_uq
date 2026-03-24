#!/bin/bash
#SBATCH --job-name=create_test_datasets
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
source /opt/ohpc/pub/apps/anaconda/3.9/2021.11/etc/profile.d/conda.sh
conda activate wlmmuq

cd /feynman/home/dap/lcs/hl285110/Documents/Code/weaklensing

srun python -u scripts/create_cropped_dataset.py -o $HOME/work/Data/kappaTNG_processed/test/LP001_cropped_384.hdf5 --idx-lp 1 --seed 42 -v

srun python -u scripts/create_cropped_dataset.py -o $HOME/work/Data/kappaTNG_processed/test/LP001_cropped_zbins_384.hdf5 --idx-lp 1 -z --seed 42 -v

srun python -u scripts/create_cropped_dataset.py -o $HOME/work/Data/kappaTNG_processed/test/LP001_cropped_brightfaint_384.hdf5 --idx-lp 1 --cosmos-include-faint --seed 42 -v

srun python -u scripts/create_cropped_dataset.py -o $HOME/work/Data/kappaTNG_processed/test/LP001_cropped_brightfaint_zbins_384.hdf5 --idx-lp 1 -z --cosmos-include-faint --seed 42 -v
