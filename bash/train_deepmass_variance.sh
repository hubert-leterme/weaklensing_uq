#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 -c deepmass_UNetPreproc_mode_wiener_nepochs_20 -a UNetPreproc -m wiener -t1 YYYYMMDD_hhmmss -e1 20 -w 8"
  exit 1
fi

optional_args="${@:2}"

# Command to execute
cmd=$(echo "python scripts/train.py ${optional_args} -uq --lr-scheduler --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
