#!/bin/bash

# Set paths
path_to_augmented_dataset=/ceph/chercheurs/leterme231/kappaTNG_augmented/LP002_augmented.hdf5
path_to_powerspectrum=/ceph/chercheurs/leterme231/kappaTNG_augmented/ps_LP002.npy
checkpoint_dir=/ceph/chercheurs/leterme231/checkpoints
save_freq=1
backup_dir=/ceph/chercheurs/leterme231/backups

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <GPU_ID> <WL_METHOD>"
  echo "Example: $0 0 ks"
  exit 1
fi

# Set environment variables and run the task
export CUDA_VISIBLE_DEVICES=$1
python scripts/train_deepmass.py $path_to_augmented_dataset \
  --input-wlmethod $2 \
  -ps $path_to_powerspectrum --lr-scheduler \
  --checkpoint-dir $checkpoint_dir/checkpoint_$2 \
  --save-freq $save_freq \
  --backup-dir $backup_dir/backup_$2 \
  --path-to-csv-log log_$2.csv --seed 42 -v
  #--nimgs-train 512 --nimgs-val 128 --nepochs 2
