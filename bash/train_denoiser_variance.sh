#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints/LP002_augmented_384

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> <NAME_DENOISER> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 denoiser_torch.SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a torch.UNetWienerInit -t1 YYYYMMDD_hhmmss -e1 100 [--scale 2.0e-1] [--scale-min 1.0e-1]"
  exit 1
fi

name_denoiser=$2
optional_args="${@:3}"

# Command to execute
cmd=$(echo "python scripts/train.py ${path_to_augmented_dataset} --denoiser --order2 ${optional_args} --lr-scheduler --checkpoint-dir ${checkpoint_dir}/${name_denoiser} --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
