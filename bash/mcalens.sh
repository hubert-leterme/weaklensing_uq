#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 -thresh 4.5 -alpha 0.5 --cqr -i 16 -w 8 --save-tensors"
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
  | grep -E '^(--nimgs-test|--nimgs-calib|--imgsize|-b|--cqr|--mode-cqr|--confidence-uq|-i|-thresh|-ig|-ing|--niter-wiener|-nw)' \
  | sed -E 's/^-b($| )/--batch-size\1/' \
  | sed -E 's/^-i($| )/--niter\1/' \
  | sed -E 's/^-thresh($| )/--starlet-detection-threshold\1/' \
  | sed -E 's/^-ig($| )/--niter-per-step-g\1/' \
  | sed -E 's/^-ing($| )/--niter-per-step-ng\1/' \
  | sed -E 's/^-nw($| )/--noise-whitening-wiener\1/' \
  | xargs \
  | sed 's/--//g' \
  | sed 's/ /_/g')
output_filename=$(echo "results_mcalens_${optional_args_cleaned}" | sed 's/__/_/g' | sed 's/_\+$//')

# Command to execute
cmd=$(echo "python scripts/pnpmass.py --mode pnpmcalens --starlet --update-ng-first ${optional_args} -o ${output_filename} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
