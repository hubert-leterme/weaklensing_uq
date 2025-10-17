#!/bin/bash
#SBATCH --mail-user hubert.leterme@cea.fr
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu_interactive
#SBATCH --gres=shard:1
#SBATCH --time=08:00:00
#SBATCH --mem=50GB
#SBATCH --output=slurm/out/jupyter.log

jupyter lab --ip="$(hostname -I|sed -e 's/.*\(10\.2\.10[45]\.[[:digit:]]*\).*/\1/')" --port=$((2**11+$RANDOM))
