#!/bin/bash

# Set paths
path_to_calibration_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP001_augmented_384.hdf5

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> <CHECKPOINT_DIR> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 checkpoint/dir/ -a torch.DRUNet -s small -t yyyymmdd_hhmmss -uq -t0 yyyymmdd_hhmmss -f 58 [-i 3] [-w 8]"
  exit 1
fi

checkpoint_dir="$2"
optional_args="${@:3}"

# Create output directory if needed
mkdir -p ${checkpoint_dir}/cqr_pnpmass

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | sed 's|--checkpoint-dir-uq [^ ]\+|alternativemn|g' \
  | sed 's/--switch-mode-for-uq//g' \
  | sed 's/--starlet -thresh /--starlet /g' \
  | sed 's/-auq [^ ]\+//g' \
  | sed 's/-suq [^ ]\+//g' \
  | sed 's/-a [^ ]\+//g' \
  | sed 's/-s [^ ]\+//g' \
  | sed 's/-t [^ ]\+//g' \
  | sed 's/-t0 [^ ]\+//g' \
  | sed 's/-w [^ ]\+//g' \
  | sed 's/-b [^ ]\+//g' \
  | sed 's/-f [^ ]\+//g' \
  | sed 's/-tau /--step-size /g' \
  | sed 's/-e0 [^ ]\+//g' \
  | sed 's/-e /--epoch /g' \
  | sed 's/-ig /--niter_g /g' \
  | sed 's/-ing /--niter_ng /g' \
  | sed 's/-i /--niter /g' \
  | sed 's/-nw /--noise-whitening-wiener /g' \
  | sed 's/-uq //g' \
  | sed 's/-ps [^ ]\+//g' \
  | sed 's/--//g' \
  | xargs \
  | sed 's/ /_/g')
output_filename=$(echo "cqr_pnpmass_${optional_args_cleaned}" | sed 's/__/_/g')

# Command to execute
cmd=$(echo "python scripts/pnpmass_calibration.py ${path_to_calibration_dataset} ${checkpoint_dir} ${optional_args} -o ${output_filename} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
