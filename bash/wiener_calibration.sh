#!/bin/bash

# Set paths
path_to_calibration_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP001_augmented_384.hdf5

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <OUTPUT_DIR> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 output/dir/ -f 58 [-w 8]"
  exit 1
fi

output_dir="$2"
optional_args="${@:3}"

# Create output directory if needed
mkdir -p ${output_dir}

# Command to execute
cmd=$(echo "python scripts/wiener_calibration.py ${path_to_calibration_dataset} ${output_dir} ${optional_args} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
