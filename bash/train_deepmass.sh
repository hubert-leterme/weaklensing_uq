#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <INPUT_METHOD> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 wiener [-m <filename> OR -a tensorflow.UNet] [--loss mse]"
  exit 1
fi

optional_args="${@:3}"

# Set name of the denoiser
optional_args_cleaned=$(echo "$optional_args" | sed 's/-m //g' | sed 's/-a //g' | sed 's/-s /--model-size /g' | sed 's/-p /--pretrained /g' | sed 's/-w [0-9]\+//g' | sed 's/-b /--batch-size /g' | sed 's/-e /--nepochs /g' | sed 's/-lr /--learning-rate /g' | sed 's/--//g' | xargs | sed 's/ /_/g' | sed 's/\.keras//g')
name_denoiser=$(echo "denoiser_${optional_args_cleaned}_${current_date}" | sed 's/__/_/g')

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --input-method $2 ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${name_denoiser} --save-freq ${save_freq} --backup-dir ${backup_dir}/${name_denoiser} --path-to-csv-log ${stats_dir}/log_${name_denoiser}_pe.csv --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
