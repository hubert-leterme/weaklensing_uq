#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> <NAME_DENOISER> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 deepmass_torch.UNetWienerInit_20250417_105243 [-a torch.UNetWienerInit] [-pred <filename> OR -o1 <timestamp>/ckp_19.pth.tar] [--loss mse]"
  exit 1
fi

model_name=$2
optional_args="${@:3}"

# Update optional arguments with full path to saved order-1 model
if [[ $optional_args == *"-o1 "* ]]; then
  order1_model_filename=$(echo "$optional_args" | xargs | grep -oP '\-o1 \K[^\s]+' | xargs)
  optional_args=$(echo "$optional_args" | sed "s|-o1 $order1_model_filename||g" | xargs)
  optional_args="-o1 ${checkpoint_dir}/${model_name}/pe/${order1_model_filename} ${optional_args}"
fi

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --wiener-init --order2 ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${model_name} --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
