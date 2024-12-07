#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <GPU_ID> <PRED_DATASET> <SCALE> <TRAINING_DATE> <LEARNING_RATE>"
  echo "Example: $0 0 LP002_augmented_pred_denoiser_1_40e_01_20241118_183322.hdf5 1.4e-1 20241118_183322 1e-4"
  exit 1
fi

pred_dataset=$2
scale=$3
training_date=$4
lr=$5
save_freq=1

scale_formatted=$(printf "%.2e" "$scale")
scale_dir=$(echo "$scale_formatted" | sed 's/[.-]/_/g')

# Set paths
datadir=/ceph/chercheurs/leterme231/kappaTNG_augmented
path_to_augmented_dataset=${datadir}/LP002_augmented.hdf5
path_to_pred_dataset=${datadir}/${pred_dataset}
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
backup_dir=/ceph/chercheurs/leterme231/backups
stats_dir=/ceph/chercheurs/leterme231/stats

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/train.py $path_to_augmented_dataset \
  --denoiser --scale $scale \
  --moment-order 2 \
  --path-to-pred-dataset $path_to_pred_dataset \
  -lr $lr --lr-scheduler \
  --checkpoint-dir $checkpoint_dir/denoiser_${scale_dir}_${training_date} \
  --save-freq $save_freq \
  --backup-dir $backup_dir/denoiser_${scale_dir}_${training_date} \
  --path-to-csv-log $stats_dir/log_denoiser_${scale_dir}_${training_date}_var.csv \
  --seed 42 -v
