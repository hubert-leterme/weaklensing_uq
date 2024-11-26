#!/bin/bash

path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
path_to_powerspectrum=/ceph/chercheurs/leterme231/kappaTNG_augmented/ps_LP002_306.npy
batch_size=256

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <GPU_ID> <WL_METHOD>"
  echo "Example: $0 0 ks"
  exit 1
fi

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/get_inputs_for_ssl.py $path_to_augmented_dataset \
  --input-wlmethod $2 \
  -ps $path_to_powerspectrum \
  -b $batch_size --seed 42 -v
