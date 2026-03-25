#!/bin/bash

#SBATCH --job-name=train_deepmass_brightfaint
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

model_name="deepmass_arch_UNetPreproc_mode-preproc_wiener_brightfaint_nepochs_20"
timestamp="20260303_182703"

srun python -u scripts/train.py -a UNetPreproc -m wiener --bin-data-from-cosmos --cosmos-include-faint -uq -t1 $timestamp -e1 20 -e 100 --lr-scheduler -c $model_name -w 8 --seed 42 -v
