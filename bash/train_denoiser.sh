#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <GPU_ID> [OPTION1 [OPTION 2 ...]]"
  echo "Example: $0 0 -a SUNetNoiseAware --scale 0.2 --scale-min 0.1 -b 16 -lr 1e-3 -w 8"
  exit 1
fi

optional_args="${@:2}"

# Set name of the denoiser
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
  | grep -E '^(-a|-s|--no-bias|-m|--additional-outlayer|-ng|--which-gaussian-extractor|-thresh|--niter-wiener|-nw|--scale|--scale-min|--nimgs-train|--nimgs-val|--imgsize|-b|--nreal-per-img|-e|-lr)' \
  | sed -E 's/^-a($| )/--arch\1/' \
  | sed -E 's/^-s($| )/--model-size\1/' \
  | sed -E 's/^-m($| )/--mode-preproc\1/' \
  | sed -E 's/^-ng($| )/--nongaussian\1/' \
  | sed -E 's/^-thresh($| )/--starlet-detection-threshold\1/' \
  | sed -E 's/^-nw($| )/--noise-whitening-wiener\1/' \
  | sed -E 's/^-b($| )/--batch-size\1/' \
  | sed -E 's/^-e($| )/--nepochs\1/' \
  | sed -E 's/^-lr($| )/--learning-rate\1/' \
  | xargs \
  | sed 's/--//g' \
  | sed 's/ /_/g')
name_denoiser=$(echo "denoiser_${optional_args_cleaned}" | sed 's/__/_/g' | sed 's/_\+$//')

# Set model specifications
optional_args_model_specs=$(echo "$optional_args" \
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
  | grep -E '^(--loss)' \
  | xargs \
  | sed 's/--//g' \
  | sed 's/ /_/g')
model_specs=$(echo "pe_${optional_args_model_specs}" | sed 's/__/_/g' | sed 's/_\+$//')

# Command to execute
cmd=$(echo "python scripts/train.py ${optional_args} -d --lr-scheduler -c ${name_denoiser} --model-specs ${model_specs} --cprofiler --cprofiler-max-nbatches 50 --cprofiler-wait 5 --cprofiler-cuda-synchronize --seed 42 -v" | xargs)

# Print the command for tracking
echo "Running the following command:"
echo "=============================================================================="
echo "$cmd"
echo "=============================================================================="
echo ""

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
eval "$cmd"
