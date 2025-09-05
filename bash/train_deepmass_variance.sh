#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints/LP002_augmented_384

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> <NAME_DENOISER> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 deepmass_torch.UNetWienerInit_nepochs_20 -a torch.UNetWienerInit -t1 YYYYMMDD_hhmmss -e1 20"
  exit 1
fi

model_name=$2
optional_args="${@:3}"

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --order2 ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${model_name} --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
