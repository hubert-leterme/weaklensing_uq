# Weak lensing mass mapping with uncertainty quantification

This repository aims at reproducing the experiments from the following papers:
- [1] *H. Leterme, J. Fadili, and J.-L. Starck, “Distribution-free uncertainty quantification for inverse problems: Application to weak lensing mass mapping,” A&A, vol. 694, p. A267, Feb. 2025;*
- [2] *H. Leterme, A. Tersenov, J. Fadili, and J.-L. Starck, “A plug-and-play approach with fast uncertainty quantification for weak lensing mass mapping,” Oct. 2025 (under review).*

**⚠️ NOTE:** This repository is currently being updated for backward compatibility with [1]. To reproduce the experiments presented in that paper, please checkout to the `distribution_free_uq` tag using `git checkout distribution_free_uq`.

## Requirements and settings

### Configuration file

Copy the configuration file `wlmmuq/config.yml` to `~/.config/wlmmuq/`, and update the values to point to the correct locations on your system for the datasets, models and other precomputed objects. This will set the global variables accordingly, in particular the paths to training / test datasets, and to saved models. Leave blank if not needed; the corresponding global variables will then be set to `None`.

### Installation

Install the `wlmmuq` library provided in this repository:
```bash
pip install .
```
All dependencies will automatically be installed.

### Virtual environments

For reproducibility, you can set one of the following virtual environments:

```bash
python -m venv wlmmuq
source wlmmuq/bin/activate
pip install -r requirements.txt
pip install . # Install wlmmuq
```

```bash
conda env create -f env.yml
conda activate wlmmuq
pip install . # Install wlmmuq
```

### Datasets

#### COSMOS shape catalog

The HST/COSMOS weak lensing reduced shear catalog from Schrabback et al. (2010, A&A 516, A63) is provided in `./data/COSMOS/`, together with some precomputed PyTorch tensors. You can move them to any location and update the configuration file accordingly.

#### The $\kappa$-TNG dataset

The $\kappa$-TNG dataset can be downloaded from the official file server—see `https://github.com/0satoken/kappaTNG` for more information. Then, data augmentation and / or cropping can be applied for ready-to-use datasets (see below).

For the sake of convenience, the power spectrum has been precomputed on $2\,048$ images from the training set, and is provided in `./data/kappaTNG/ps_LP002_384.pt`. You can move this file to any location and update the configuration file accordingly.

## Python scripts

### Creating ready-to-use datasets

- **Training / validation and calibration sets:** Data augmentation by rotating and randomly cropping convergence maps from the $\kappa$-TNG dataset;
- **Test set:** Obtained by cropping original data to the desired size.

| | Training | Validation | Calibration | Test |
|-|----------|------------|-------------|------|
| Nb images | $70\,560$ | $1\,440$ | $1\,935$ | $513$
| Lensing potential | LP002 | LP002 | LP001 | LP001 |
| Independent realizations | 001 → 098 | 099 → 100 | 058 → 100 | 001 → 057 |
| Type of augmentation | Rotations & Crops | Rotations & Crops | Rotations & Crops | Center crops |
| Nb rotation angles | $360$ | $360$ | $45$ | N/A |
| Nb random crops per angle | $2$ | $2$ | $1$ | N/A |
| Nb crops per realization | N/A | N/A | N/A | $3 \times 3$ |

#### Training & validation sets

```bash
python scripts/create_augmented_dataset.py -o path/to/training/and/validation/dataset.hdf5 --idx-lp 2 --angle-step 1 --niter-per-angle 2 --seed 42 -v
```

The training and validation samples are stored together in a single HDF5 file. The separation between the two sets is performed dynamically when loading the data.

#### Calibration & test sets

```bash
python scripts/create_augmented_dataset.py -o path/to/calib/dataset.hdf5 --idx-lp 1 --angle-step 8 --seed 42 -v
python scripts/create_cropped_dataset.py -o path/to/test/dataset.hdf5 --idx-lp 1 --seed 42 -v
```

Both datasets are generated from the full set of 100 independent realizations corresponding to the specified lensing potential. To prevent any overlap between the calibration and test samples, the data should be filtered by realization index when loading the datasets (see the table above for the specific realization ranges used for each set).

### Training denoisers for PnPMass

#### Point estimate (order-1 networks)

```bash
# Standard version
python scripts/train.py -a SUNetNoiseAware -d --scale 0.2 --scale-min 0.0 -b 16 -e 100 -lr 1e-3 --lr-scheduler -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v

# Residual version
python scripts/train.py -a SUNetNoiseAware -d -ng --scale 0.2 --scale-min 0.0 -b 16 -e 100 -lr 1e-3  --lr-scheduler -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v
```

#### Variance estimate (order-2 networks)

```bash
# Standard version
python scripts/train.py -a SUNetNoiseAware -d -uq -t1 YYYYMMDD_hhmmss -e1 100 --scale 0.2 --scale-min 0.0 -b 16 -e 100 -lr 1e-3 --lr-scheduler -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v

# Residual version
python scripts/train.py -a SUNetNoiseAware -d -ng -uq -t1 YYYYMMDD_hhmmss -e1 100 --scale 0.2 --scale-min 0.0 -b 16 -e 100 -lr 1e-3 --lr-scheduler -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -w 8 --seed 42 -v
```

The optional argument `-t1 YYYYMMDD_hhmmss` must be replaced by the timestamp of the previously-trained order-1 model. It corresponds to the folder name containing the saved checkpoints (e.g., `~/model/dir/denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3/pe/20250613_143319/ckp_100.pth.tar`). The folder name `pe` stands for "point estimate" (order-1 models), whereas `var` stands for "variance" (order-2 models).

### Training DeepMass

#### Point estimate (order-1 networks)

```bash
# COSMOS "bright" catalog only
# Noise standard deviation and mask have been pre-computed and stored in `wlmmuq.PATH_TO_STD_NOISE` and `wlmmuq.PATH_TO_MASK`.
# Otherwise, use CLI argument `--bin-data-from-cosmos`
python scripts/train.py -a UNetPreproc -m wiener --bin-data-from-cosmos -e 20 --lr-scheduler -c deepmass_arch_UNetPreproc_mode-preproc_wiener_nepochs_20 -w 8 --seed 42 -v

# COSMOS "bright" + "faint" catalogs
# Recompute noise standard deviation and mask using the CLI arguments `--bin-data-from-cosmos` and `--cosmos-include-faint`
python scripts/train.py -a UNetPreproc -m wiener --bin-data-from-cosmos --cosmos-include-faint -e 20 --lr-scheduler -c deepmass_arch_UNetPreproc_mode-preproc_wiener_brightfaint_nepochs_20 -w 8 --seed 42 -v
```

#### Variance estimate (order-2 networks)

```bash
# COSMOS "bright" catalog only 
python scripts/train.py -a UNetPreproc -m wiener --bin-data-from-cosmos -uq -t1 YYYYMMDD_hhmmss -e1 20 -e 100 --lr-scheduler -c deepmass_arch_UNetPreproc_mode-preproc_wiener_nepochs_20 -w 8 --seed 42 -v

# COSMOS "bright" + "faint" catalogs
python scripts/train.py -a UNetPreproc -m wiener --bin-data-from-cosmos --cosmos-include-faint -uq -t1 YYYYMMDD_hhmmss -e1 20 -e 100 --lr-scheduler -c deepmass_arch_UNetPreproc_mode-preproc_wiener_brightfaint_nepochs_20 -w 8 --seed 42 -v
```

The order-2 network has been trained on 100 epochs, vs only 20 epochs for the order-1 network. This was motivated by the validation loss still decreasing after $20$ epochs.

### Running PnPMass

#### Test several step sizes

The following scripts run PnPMass with a step size set to 50%, 75% and 100% of its upper limit.

```bash
# Standard version
python scripts/pnpmass.py -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250613_143319 -alph 0.5 0.75 1.0 -i 8 -w 8 -o niter_8 --save-tensors --nimgs-save 8 --seed 42 -v

# Residual version
python scripts/pnpmass.py -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250716_170944 --mode residual -alph 0.5 0.75 1.0 -i 8 -w 8 -o mode_residual_niter_8 --save-tensors --nimgs-save 8 --seed 42 -v
```

#### Run PnPMass with uncertainty quantification and conformal prediction

```bash
# Standard version
python scripts/pnpmass.py -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250613_143319 -uq -t0 20250903_164013 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o niter_8_cqr --seed 42 -v

# Residual version
python scripts/pnpmass.py -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250716_170944 --mode residual -uq -t0 20250903_164205 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o mode_residual_niter_8_cqr --seed 42 -v
```
To recompute the noise standard deviation and mask from the COSMOS shape catalog, add CLI argument `--bin-data-from-cosmos`, in combination with `--cosmos-include-faint` to use both the bright and faint catalogs.

#### Run PnPMass on the COSMOS shear map

```bash
# Standard version
python scripts/pnpmass.py --bin-data-from-cosmos --test-on-real-data -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250613_143319 -uq -t0 20250903_164013 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o cosmos_niter_8_cqr --seed 42 -v

# Residual version
python scripts/pnpmass.py --bin-data-from-cosmos --test-on-real-data -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 -a SUNetNoiseAware -t 20250716_170944 --mode residual -uq -t0 20250903_164205 -i 8 --cqr --find-optimal-hyperparam-precalib -w 8 --save-tensors -o cosmos_mode_residual_niter_8_cqr --seed 42 -v
```
To use both the bright and faint catalogs, add CLI argument `--cosmos-include-faint`. To run PnPMass on both the simulated test set and the real COSMOS shear map, use CLI argument `--run-both` instead of `--test-on-real-data`.

### Other mass mapping methods

Available Python scripts for the following methods:
- DeepMass – `scripts/deepmass.py`;
- MCALens – `scripts/mcalens.py`;
- Iterative Wiener filtering – `scripts/wiener.py`;
- Kaiser-Squires filtering – `scripts/ks.py`.

These scripts follow the same logic as `scripts/pnpmass.py`. Check the related documentation for more details.

## Jupyter notebooks

Examples are provided in the Jupyter notebooks provided in the directory `./notebooks`.

## License

Copyright 2025 Hubert Leterme & Andreas Tersenov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
