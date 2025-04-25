#!/bin/bash

path_to_test_dataset=/home/leterme231/Documents/Data/kappaTNG_cropped/LP001_cropped.hdf5

current_date=$(date +"%Y%m%d_%H%M%S")

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <METHOD> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 mcalens --niter 100 --Nsigma 5"
  exit 1
fi

method=$1
optional_args="${@:2}"

# Set name of the saved array
optional_args_cleaned=$(echo "$optional_args" | sed 's/-b [0-9]\+//g' | sed 's/-w [0-9]\+//g' | sed 's/--//g' | xargs | sed 's/ /_/g')
picklename=$(echo "${method}_${optional_args_cleaned}_${current_date}" | sed 's/__/_/g')

cmd=$(echo "python scripts/massmapping.py $method $picklename $path_to_test_dataset $optional_args --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

eval "$cmd"
