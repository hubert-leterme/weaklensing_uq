#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <GPU_ID> <SCALE> <TRAINED_MODEL> <OUTPUT_DATASET>"
  echo "Example: $0 0 1.4e-1 denoiser_1_40e_01_20241118_183322/pe/20.keras LP002_augmented_pred_denoiser_1_40e_01_20241118_183322.hdf5"
  exit 1
fi

scale=$2
train_model=$3
output_dataset=$4

# Set paths
path_to_trained_model=/ceph/chercheurs/leterme231/checkpoints/$train_model
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
path_to_output_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/$output_dataset

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/predict.py \
  $path_to_trained_model $path_to_augmented_dataset $path_to_output_dataset \
  --denoiser --scale $scale \
  --seed 42 -v
