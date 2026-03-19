#!/bin/bash

# PnPMass, standard version
python scripts/pnpmass.py --run-both -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250613_143319 -uq -t0 20250903_164013 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o inputs-from-mmgan_niter_8_cqr --seed 42 -v

# PnPMass, residual version
python scripts/pnpmass.py --run-both -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250716_170944 --mode residual -uq -t0 20250903_164205 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o inputs-from-mmgan_niter_8_cqr --seed 42 -v

# DeepMass
python scripts/deepmass.py --run-both -c deepmass_arch_UNetPreproc_mode-preproc_wiener_nepochs_20 -a UNetPreproc -m wiener -t 20250613_142121 -e 20 -uq -t0 20250917_153127 -e0 100 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o inputs-from-mmgan_epoch_20_epoch-uq_100_cqr --seed 42 -v

# MCALens
python scripts/mcalens.py --run-both -thresh 4.5 -alph 0.5 -i 16 --cqr -w 8 --save-tensors -o inputs-from-mmgan_starlet-detection-threshold_4.5_niter_16_cqr --seed 42 -v

# Wiener
python scripts/wiener.py --run-both --cqr -w 8 --save-tensors -o inputs-from-mmgan_cqr --seed 42 -v

# Kaiser-Squires
python scripts/ks.py --run-both --cqr -w 8 --save-tensors -o inputs-from-mmgan_cqr --seed 42 -v
