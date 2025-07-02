#!/bin/bash

# Set paths
path_to_test_dataset=/ceph/chercheurs/leterme231/kappaTNG_cropped/LP001_cropped_384.hdf5

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <CHECKPOINT_DIR> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 checkpoint/dir/ -cqr path/to/cqr.pt -a torch.DRUNet -s small -t yyyymmdd_hhmmss -uq -t0 yyyymmdd_hhmmss [-i 3] [-w 8]"
  exit 1
fi

checkpoint_dir="$2"
optional_args="${@:3}"

# Create output directory if needed
mkdir -p ${checkpoint_dir}/results_pnpmass

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | sed 's/-cqr [^ ]\+//g' \
  | sed 's/-a [^ ]\+//g' \
  | sed 's/-s [^ ]\+//g' \
  | sed 's/-t [^ ]\+//g' \
  | sed 's/-e [^ ]\+//g' \
  | sed 's/-t0 [^ ]\+//g' \
  | sed 's/-w [^ ]\+//g' \
  | sed 's/-b [^ ]\+//g' \
  | sed 's/-f [^ ]\+//g' \
  | sed -E 's/-tau( [^-][^ ]*)+//g' \
  | sed 's/-i /--niter /g' \
  | sed 's/-uq //g' \
  | sed 's/--save-tensors //g' \
  | sed 's/--//g' \
  | xargs \
  | sed 's/ /_/g')
output_filename=$(echo "results_pnpmass_${optional_args_cleaned}" | sed 's/__/_/g')
path_to_output=$(echo "${checkpoint_dir}/results_pnpmass/${output_filename}" | sed 's|//|/|g' | xargs)

# Command to execute
cmd=$(echo "python scripts/pnpmass.py ${path_to_test_dataset} ${checkpoint_dir} ${path_to_output} ${optional_args} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
