#!/bin/bash

nimgs_test=225 # Evaluate on the same dataset as KS, Wiener and MCALens
batch_size=45

# Set paths
path_to_test_set=/ceph/chercheurs/leterme231/kappaTNG_cropped/LP001_cropped.hdf5
path_to_model=/ceph/chercheurs/leterme231/checkpoints/checkpoint_$2/20.keras
path_to_output=/ceph/chercheurs/leterme231/eval/$2.pickle
path_to_powerspectrum=/ceph/chercheurs/leterme231/kappaTNG_augmented/ps_LP002.npy

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <GPU_ID> <WL_METHOD>"
  echo "Example: $0 0 ks"
  exit 1
fi

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/evaluate_deepmass.py $path_to_test_set $path_to_model $path_to_output \
  --input-wlmethod $2 \
  -ps $path_to_powerspectrum \
  --nimgs-test $nimgs_test \
  -b $batch_size \
  --seed 42 -v
