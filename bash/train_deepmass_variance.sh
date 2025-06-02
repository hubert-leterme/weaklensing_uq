#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented_384.hdf5
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1

# Check if correct number of arguments are provided
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <GPU_ID> <WL_METHOD> <PRED_DATASET> <TRAINING_DATE> <LEARNING_RATE>"
  echo "Example: $0 0 ks /path/to/pred/dataset.hdf5 20241105_213110 1e-4"
  exit 1
fi

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/train.py $path_to_augmented_dataset \
  --input-method $2 \
  --moment-order 2 \
  --path-to-pred-dataset $3 \
  -lr $5 --lr-scheduler \
  --checkpoint-dir $checkpoint_dir/checkpoint_${2}_${4} \
  --seed 42 -v
