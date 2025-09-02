#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints/LP002_augmented_384

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 [-a torch.DRUNet] [--scale 2.0e-1] [--scale-min 1.0e-1] [--loss mse]"
  exit 1
fi

optional_args="${@:2}"

# Set name of the denoiser
optional_args_cleaned=$(echo "$optional_args" \
  | sed 's/--timestamp-resume [^ ]\+//g' \
  | sed 's/--epoch-resume [^ ]\+//g' \
  | sed 's/-a //g' \
  | sed 's/-thresh /--starlet-detection-threshold /g' \
  | sed 's/-s /--model-size /g' \
  | sed 's/-p /--pretrained /g' \
  | sed 's/-w [^ ]\+//g' \
  | sed 's/-b /--batch-size /g' \
  | sed 's/-e /--nepochs /g' \
  | sed 's/-lr /--learning-rate /g' \
  | sed 's/-r //g' \
  | sed 's/--//g' \
  | xargs \
  | sed 's/ /_/g')
name_denoiser=$(echo "denoiser_${optional_args_cleaned}" | sed 's/__/_/g')

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --denoiser ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${name_denoiser} --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
