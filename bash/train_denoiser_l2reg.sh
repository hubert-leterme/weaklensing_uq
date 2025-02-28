#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1
backup_dir=/ceph/chercheurs/leterme231/backups
stats_dir=/ceph/chercheurs/leterme231/stats

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <GPU_ID> <SCALE> <LEARNING_RATE> <LOSS> <L2_LAMBDA>"
  echo "Example: $0 0 1.4e-1 1e-4 l2reg_mse 1e-4"
  exit 1
fi

scale=$2
lr=$3
loss=$4
l2_lambda=$5

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/train.py $path_to_augmented_dataset \
  --denoiser --scale $scale --scale-range \
  -lr $lr --lr-scheduler \
  --loss $loss --l2-lambda $l2_lambda \
  --checkpoint-dir $checkpoint_dir/denoiser_${scale}_${loss}_${l2_lambda}_${current_date} \
  --save-freq $save_freq \
  --backup-dir $backup_dir/denoiser_${scale}_${loss}_${l2_lambda}_${current_date} \
  --path-to-csv-log $stats_dir/log_denoiser_${scale}_${loss}_${l2_lambda}_${current_date}_pe.csv --seed 42 -v
