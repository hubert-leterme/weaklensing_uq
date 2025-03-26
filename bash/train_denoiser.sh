#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1
backup_dir=/ceph/chercheurs/leterme231/backups
stats_dir=/ceph/chercheurs/leterme231/stats

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 [--scale 2.0e-1] [--scale-min 1.0e-1] [--loss l2reg_mse]"
  exit 1
fi

optional_args="${@:4}"

# Process optional arguments: remove leading '--' or '-'
optional_args_cleaned=$(echo "$optional_args" | sed 's/--//g' | sed 's/ /_/g')

name_denoiser="denoiser_${optional_args_cleaned}_${current_date}"

# Command to execute
cmd="python scripts/train.py ${path_to_augmented_dataset} --denoiser ${optional_args} -lr 1e-4 --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${name_denoiser} --save-freq ${save_freq} --backup-dir ${backup_dir}/${name_denoiser} --path-to-csv-log ${stats_dir}/log_${name_denoiser}_pe.csv --seed 42 -v"

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
