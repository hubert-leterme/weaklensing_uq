#!/bin/bash

# Set paths
path_to_test_dataset=/ceph/checheurs/leterme231/Data/kappaTNG_cropped/LP001_cropped_384.hdf5
path_to_ps=/ceph/chercheurs/leterme231/Data/kappaTNG_augmented/ps_LP002_384.pt

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <PATH_TO_CHECKPOINT> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 path/to/checkpoint/pe/yyyymmdd_hhmmss/ckp_100.pth.tar [-a torch.DRUNet -s small] [-i 3] [-w 8]"
  exit 1
fi

checkpoint_dir="$2"
optional_args="${@:3}"

# Set output filename
path_to_output="${checkpoint_dir}/results_deepmass"

# Command to execute
cmd=$(echo "python scripts/deepmass.py ${path_to_test_dataset} ${checkpoint_dir} ${path_to_ps} ${path_to_output} ${optional_args} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
