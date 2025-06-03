#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
path_to_models=/ceph/chercheurs/leterme231/models
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 [-m <filename> OR -a torch.DRUNet] [--scale 2.0e-1] [--scale-min 1.0e-1] [--loss mse]"
  exit 1
fi

optional_args="${@:2}"

# Set name of the denoiser
optional_args_cleaned=$(echo "$optional_args" | sed 's/-m //g' | sed 's/-a //g' | sed 's/-s /--model-size /g' | sed 's/-p /--pretrained /g' | sed 's/-w [0-9]\+//g' | sed 's/-b /--batch-size /g' | sed 's/-e /--nepochs /g' | sed 's/-lr /--learning-rate /g' | sed 's/--//g' | xargs | sed 's/ /_/g')
name_denoiser=$(echo "denoiser_${optional_args_cleaned}_${current_date}" | sed 's/__/_/g')

# Check if argument `-m <model_filename>` is provided and
# update optional arguments with full path to model
if [[ $optional_args == *"-m "* ]]; then
  model_filename=$(echo "$optional_args" | xargs | grep -oP '\-m \K[^\s]+' | xargs)
  optional_args=$(echo "$optional_args" | sed "s/-m $model_filename//g" | xargs)
  optional_args="-m ${path_to_models}/${model_filename} ${optional_args}"
fi

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --denoiser ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${name_denoiser} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
