#!/bin/bash

#SBATCH --job-name=train_denoiser_zbins
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4g.20gb
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --output /feynman/home/dap/lcs/hl285110/work/Log/slurm_out/out_%j.log

# Load conda and activate environment
source /opt/ohpc/pub/apps/anaconda/3.9/2021.11/etc/profile.d/conda.sh
conda activate wlmmuq

cd /feynman/home/dap/lcs/hl285110/Documents/Code/weaklensing

nvidia-smi

srun python -u scripts/train.py -a SUNetNoiseAware -d --scale 0.3 --scale-min 0.0 --path-to-train-val-dataset $HOME/work/Data/kappaTNG_processed/trainval/LP002_augmented_zbins_384.hdf5 -b 16 -e 100 -lr 1e-3 --lr-scheduler -c denoiser_arch_SUNetNoiseAware_zbins_scale_0.3_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v
