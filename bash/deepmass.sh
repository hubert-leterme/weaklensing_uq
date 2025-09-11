#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 -a torch.UNetPreproc -m wiener -t yyyymmdd_hhmmss -e20 -uq -auq torch.UNetPreproc -muq ks -t0 yyyymmdd_hhmmss --cqr -w 8"
  exit 1
fi

optional_args="${@:2}"

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | xargs -n1 \
  | awk '
    /^-/ {
      if (NR > 1) printf "\n";
      printf "%s", $0;
      next
    }
    { printf " %s", $0 }
    END { printf "\n" }
  ' \
  | grep -E '^(-e|-c0|-e0|--nimgs-test|--nimgs-calib|--imgsize|-b|--cqr|--mode-cqr|--confidence-uq|--niter-wiener|-nw)' \
  | sed -E 's/^-e($| )/--epoch\1/' \
  | sed 's/-c0 [^ ]\+/alternativemn/g' \
  | sed -E 's/^-e0($| )/--epoch-uq\1/' \
  | sed -E 's/^-b($| )/--batch-size\1/' \
  | sed -E 's/^-nw($| )/--noise-whitening-wiener\1/' \
  | xargs \
  | sed 's/--//g' \
  | sed 's/ /_/g')
output_filename=$(echo "results_deepmass_${optional_args_cleaned}" | sed 's/__/_/g' | sed 's/_\+$//')

# Command to execute
cmd=$(echo "python scripts/deepmass.py ${optional_args} -o ${output_filename} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
