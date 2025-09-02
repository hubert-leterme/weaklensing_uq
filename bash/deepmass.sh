#!/bin/bash

# Set paths
path_to_test_dataset=/ceph/chercheurs/leterme231/kappaTNG_cropped/LP001_cropped_384.hdf5

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <CHECKPOINT_DIR> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 checkpoint/dir/ -cqr path/to/cqr.pt -a torch.DRUNet -s small -t yyyymmdd_hhmmss -uq -t0 yyyymmdd_hhmmss [-w 8]"
  exit 1
fi

checkpoint_dir="$2"
optional_args="${@:3}"

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | sed 's|--checkpoint-dir-uq [^ ]\+|alternativemn|g' \
  | sed 's/-auq [^ ]\+//g' \
  | sed 's/-suq [^ ]\+//g' \
  | sed 's/-cqr [^ ]\+//g' \
  | sed 's/-a [^ ]\+//g' \
  | sed 's/-s [^ ]\+//g' \
  | sed 's/-t [^ ]\+//g' \
  | sed 's/-t0 [^ ]\+//g' \
  | sed 's/-w [^ ]\+//g' \
  | sed 's/-b [^ ]\+//g' \
  | sed 's/-f [^ ]\+//g' \
  | sed 's/-e0 [^ ]\+//g' \
  | sed 's/-e /--epoch /g' \
  | sed 's/-uq //g' \
  | sed 's/-ps [^ ]\+//g' \
  | sed 's/--//g' \
  | xargs \
  | sed 's/ /_/g')
output_filename=$(echo "results_deepmass_${optional_args_cleaned}" | sed 's/__/_/g' | sed 's/_\+$//')

# Command to execute
cmd=$(echo "python scripts/deepmass.py ${path_to_test_dataset} ${checkpoint_dir} ${optional_args} -o ${output_filename} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
