#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1
backup_dir=/ceph/chercheurs/leterme231/backups
stats_dir=/ceph/chercheurs/leterme231/stats

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <GPU_ID> <SCALE> <LEARNING_RATE>"
  echo "Example: $0 0 7e-2 1e-4"
  exit 1
fi

scale_formatted=$(printf "%.2e" "$2")
scale_dir=$(echo "$scale_formatted" | sed 's/[.-]/_/g')

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/train.py $path_to_augmented_dataset \
  --denoiser --scale-denoiser $2 \
  -lr $3 --lr-scheduler \
  --checkpoint-dir $checkpoint_dir/denoiser_${scale_dir}_${current_date} \
  --save-freq $save_freq \
  --backup-dir $backup_dir/denoiser_${scale_dir}_${current_date} \
  --path-to-csv-log $stats_dir/log_denoiser_${scale_dir}_${current_date}_pe.csv --seed 42 -v
