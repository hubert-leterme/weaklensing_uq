#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 -c subdir -a torch.SUNetNoiseAware -t yyyymmdd_hhmmss -uq -t0 yyyymmdd_hhmmss -i 8 --cqr -w 8 --save-tensors"
  exit 1
fi

optional_args="${@:2}"

# Set output filename
optional_args_cleaned=$(echo "$optional_args" \
  | xargs -n1 printf '%s\n' \
  | awk '
    /^-/ {
      if (NR > 1) printf "\n";
      printf "%s", $0;
      next
    }
    { printf " %s", $0 }
    END { printf "\n" }
  ' \
  | grep -E '^(-e|-c0|-e0|--nimgs-test|--nimgs-calib|--imgsize|-b|--cqr|--mode-cqr|--confidence-uq|-i|--mode|--which-gaussian-extractor|--update-ng-first|--starlet|-thresh|-ig|-ing|--niter-wiener|-nw|--multfact-step-size-gaussian)' \
  | sed -E 's/^-e($| )/--epoch\1/' \
  | sed 's/-c0 [^ ]\+/alternativemn/g' \
  | sed -E 's/^-e0($| )/--epoch-uq\1/' \
  | sed -E 's/^-b($| )/--batch-size\1/' \
  | sed -E 's/^-i($| )/--niter\1/' \
  | sed -E 's/^-thresh($| )/--starlet-detection-threshold\1/' \
  | sed -E 's/^-ig($| )/--niter-per-step-g\1/' \
  | sed -E 's/^-ing($| )/--niter-per-step-ng\1/' \
  | sed -E 's/^-nw($| )/--noise-whitening-wiener\1/' \
  | xargs \
  | sed 's/--//g' \
  | sed 's/ /_/g')
output_filename=$(echo "results_pnpmass_${optional_args_cleaned}" | sed 's/__/_/g' | sed 's/_\+$//')

# Command to execute
cmd=$(echo "python scripts/pnpmass.py ${optional_args} -o ${output_filename} --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
