#!/bin/bash
#SBATCH --job-name=get_powerspectrum
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.10gb
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
source /opt/ohpc/pub/apps/anaconda/3.9/2021.11/etc/profile.d/conda.sh
conda activate wlmmuq

cd /feynman/home/dap/lcs/hl285110/Documents/Code/weaklensing

echo "Compute power spectrum for the bright catalog only; one redshift bin"
srun python -u scripts/get_powerspectrum.py --path-to-train-dataset $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_384.hdf5 -o $HOME/work/Data/kappaTNG_processed/trainval/ps_LP002_384.pt -w 8 --seed 42 -v

echo "Compute power spectrum for the bright catalog only; several redshift bins"
srun python -u scripts/get_powerspectrum.py --path-to-train-dataset $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_zbins_384.hdf5 -o $HOME/work/Data/kappaTNG_processed/trainval/ps_LP002_zbins_384.pt -w 8 --seed 42 -v

echo "Compute power spectrum for the bright + faint catalogs; one redshift bin"
srun python -u scripts/get_powerspectrum.py --path-to-train-dataset $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_brightfaint_384.hdf5 -o $HOME/work/Data/kappaTNG_processed/trainval/ps_LP002_brightfaint_384.pt -w 8 --seed 42 -v

echo "Compute power spectrum for the bright + faint catalogs; several redshift bins"
srun python -u scripts/get_powerspectrum.py --path-to-train-dataset $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_brightfaint_zbins_384.hdf5 -o $HOME/work/Data/kappaTNG_processed/trainval/ps_LP002_brightfaint_zbins_384.pt -w 8 --seed 42 -v
