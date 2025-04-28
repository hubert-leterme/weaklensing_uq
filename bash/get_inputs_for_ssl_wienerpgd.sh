#!/bin/bash

path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
path_to_powerspectrum=/ceph/chercheurs/leterme231/kappaTNG_augmented/ps_LP002_306.npy
batch_size=256

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <GPU_ID> <STEP_SIZE> <NITER>"
  echo "Example: $0 0 1.96e-2 5"
  exit 1
fi

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/get_inputs_for_ssl.py $path_to_augmented_dataset \
  --input-method wiener_pgd \
  -ps $path_to_powerspectrum \
  --step-size $2 --niter $3 \
  -b $batch_size --seed 42 -v
