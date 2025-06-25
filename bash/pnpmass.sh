#!/bin/bash

# Set paths
path_to_test_dataset=/ceph/chercheurs/leterme231/kappaTNG_cropped/LP001_cropped_384.hdf5

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <PATH_TO_CHECKPOINT> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 path/to/checkpoint/pe/yyyymmdd_hhmmss/ckp_100.pth.tar [-a torch.DRUNet -s small] [-i 3] [-w 8]"
  exit 1
fi

path_to_checkpoint="$2"
optional_args="${@:3}"

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | sed 's/-a [^ ]\+//g' \
  | sed 's/-s [^ ]\+//g' \
  | sed 's/-w [^ ]\+//g' \
  | sed 's/-tau /--step-size /g' \
  | sed 's/-i /--niter /g' \
  | sed 's/-b /--batch-size /g' \
  | sed 's/--//g' \
  | xargs \
  | sed 's/ /_/g')
output_filename=$(echo "results_pnpmass_${optional_args_cleaned}" | sed 's/__/_/g')

# Apply `dirnames` three times to get the output directory
output_dir=$(dirname "$(dirname "$(dirname "$path_to_checkpoint")")")
path_to_output="${output_dir}/${output_filename}"

# Command to execute
cmd=$(echo "python scripts/pnpmass.py ${path_to_test_dataset} ${path_to_checkpoint} ${path_to_output} ${optional_args} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
