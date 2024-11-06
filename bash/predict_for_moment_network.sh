#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <GPU_ID> <WL_METHOD> <TRAINED_MODEL> <OUTPUT_DATASET>"
  echo "Example: $0 0 ks checkpoint_ks/20.keras LP002_augmented_pred_ks.hdf5"
  exit 1
fi

# Set paths
path_to_trained_model=/ceph/chercheurs/leterme231/checkpoints/$3
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
path_to_output_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/$4
path_to_powerspectrum=/ceph/chercheurs/leterme231/kappaTNG_augmented/ps_LP002.npy

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/predict_for_moment_network.py \
  $path_to_trained_model $path_to_augmented_dataset $path_to_output_dataset \
  --input-wlmethod $2 \
  -ps $path_to_powerspectrum \
  --seed 42 -v
