#!/bin/bash
#SBATCH --mail-user hubert.leterme@cea.fr
#SBATCH --job-name=train_denoiser_zbins
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.10gb
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --output slurm/out/out_%j.log

python scripts/train.py -a SUNetNoiseAware -d --scale 0.3 --scale-min 0.0 -b 16 -e 100 -lr 1e-3 --lr-scheduler -c denoiser_arch_SUNetNoiseAware_zbins_scale_0.3_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v
