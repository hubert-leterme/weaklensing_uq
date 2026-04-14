# Copilot Instructions for weaklensing_uq

## Project Overview

This repository implements weak lensing mass mapping with uncertainty quantification. It includes:
- **Core Library** (`wlmmuq/`): Python package for models, datasets, training, and inference
- **Scripts** (`scripts/`): CLI tools for data augmentation, training denoisers, and running mass mapping algorithms
- **Notebooks** (`notebooks/`): Jupyter notebooks demonstrating workflows

The project reproduces experiments from academic papers on distribution-free uncertainty quantification for inverse problems in weak lensing.

## Installation & Setup

### Install the Package

```bash
pip install -e . --config-settings editable_mode=compat
```

Or with dependencies explicitly:

```bash
pip install -r requirements.txt
pip install -e .
```

### Configuration

1. Copy `wlmmuq/config.yml` to `~/.config/wlmmuq/config.yml`
2. Update paths to point to your local datasets and model directories:
   - `cosmos_dir`: HST/COSMOS weak lensing catalog
   - `ktng_dir`: κ-TNG simulation dataset
   - `model_dir`: Location to save trained models
   - `results_dir`: Location to save inference results
   - Data paths: `path_to_train_val_dataset`, `path_to_test_dataset`, `path_to_calib_dataset`
   - Precomputed objects: `path_to_ps`, `path_to_std_noise`, `path_to_mask`, `path_to_real_shearmap`

The config loader searches in this order: current directory → `~/.config/wlmmuq/` → `/etc/wlmmuq/` → package default.

### Virtual Environments

**venv:**
```bash
python -m venv wlmmuq
source wlmmuq/bin/activate
pip install -r requirements.txt
pip install -e .
```

**conda:**
```bash
conda env create -f env.yml
conda activate wlmmuq
pip install -e .
```

## Architecture & Key Modules

### Core Modules (`wlmmuq/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Global configuration from YAML; exports paths and settings as module-level variables |
| `datasets/` | Data loading and augmentation (COSMOS, κ-TNG); base dataset classes handle HDF5 I/O |
| `models/` | Neural network architectures (SUNet, UNet) and algorithms (PnPMass, DeepMass, Wiener, Kaiser-Squires) |
| `loss/` | Loss functions and metrics (MSE, MAE) for model training |
| `optim/` | Optimization iterators and algorithms (MCALens, proximal methods) |
| `physics.py` | Weak lensing physics utilities (shear/convergence relations) |
| `transform.py` | Data transformations (normalization, scaling) |
| `training.py` | Training loops using PyTorch Lightning |
| `callbacks.py` | Training callbacks (logging, checkpointing) |
| `utils.py` | Utility functions |

### Script Patterns (`scripts/`)

Scripts use a consistent structure via `_commons.py` and `_add_arguments.py`:

1. **Entry point**: `main()` function with full default parameters from `wlmmuq` config
2. **CLI arguments**: Created by shared argument builders that inherit from config defaults
3. **Dataset selection**: Training vs test/calibration sets controlled by config paths
4. **Model/method selection**: Architecture (-a), mode, and checkpoint timestamp (-t) flags
5. **Output naming**: Automatically based on method, model config, and optional custom suffix (-o)

**Key script categories:**
- `create_augmented_dataset.py`, `create_cropped_dataset.py`: Data preparation
- `train.py`: Train denoisers (SUNetNoiseAware) with optional UQ (variance estimates)
- `pnpmass.py`, `deepmass.py`, `mcalens.py`, `wiener.py`, `ks.py`: Mass mapping methods

### Dataset Structure

Datasets split training/validation dynamically. All sets use HDF5 format. Key details:

| Aspect | Training | Validation | Calibration | Test |
|--------|----------|------------|-------------|------|
| **Nb images** | 70,560 | 1,440 | 1,935 | 513 |
| **Lensing potential** | LP002 | LP002 | LP001 | LP001 |
| **Independent realizations** | 001–098 | 099–100 | 058–100 | 001–057 |
| **Type of augmentation** | Rotations & Crops | Rotations & Crops | Rotations & Crops | Center crops |
| **Nb rotation angles** | 360 | 360 | 45 | N/A |
| **Nb random crops per angle** | 2 | 2 | 1 | N/A |
| **Nb crops per realization** | N/A | N/A | N/A | 3 × 3 |

- **Train/Val**: Stored together in single HDF5 file; separation performed dynamically when loading
- **Test**: 512 center-cropped images from LP001 realizations 001–057
- **Calibration**: 1,935 augmented images from LP001 realizations 058–100

Use `--idx-lp` to select lensing potential (001-100). COSMOS catalogs available (bright/faint).

## Key Conventions

### Config Resolution Pattern

```python
import wlmmuq as wl

# Access config values directly (returns None if not set)
model_dir = wl.MODEL_DIR
dataset_path = wl.PATH_TO_TEST_DATASET
```

Configuration is read at import time, so modify `~/.config/wlmmuq/config.yml` before importing.

### Model Naming

Model checkpoint directories follow this pattern:
```
{model_dir}/{method_name}_{arch}_{config_params}/{mode}/{timestamp}/ckp_{epoch}.pth.tar
```

- `method_name`: e.g., "denoiser" or "deepmass"
- `arch`: Network architecture (e.g., "SUNetNoiseAware", "UNetPreproc")
- `mode`: "pe" (point estimate, order-1) or "var" (variance, order-2)
- `timestamp`: YYYYMMDD_hhmmss (directory name for checkpoints)

**Example**: `~/.model/dir/denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3/pe/20250613_143319/ckp_100.pth.tar`

### Training Modes

- **Point estimate (PE, order-1)**: Trains single network to predict convergence
- **Variance estimate (order-2)**: Second network trained from fixed PE checkpoint to predict variance
  - Pass `-t1 YYYYMMDD_hhmmss` (PE checkpoint timestamp) and `-e1 PE_EPOCHS`
  - Mode stored as "var" in output path

### Uncertainty Quantification

- `-uq` flag enables variance output
- `-t0 YYYYMMDD_hhmmss` selects calibration model checkpoint
- `--cqr` enables Conformal Quantile Regression
- `--find-optimal-hyperparam-precalib` auto-tunes CQR hyperparameter on calibration set

### Dataset Arguments

Shared across most scripts:
- `-o`, `--path-to-output`: Output file path
- `--idx-lp`: Learning potential index (001-100, default varies by script)
- `-b`, `--batch-size`: Batch size for I/O (prevents OOM; not training batch size)
- `-w`, `--num-workers`: PyTorch DataLoader workers
- `--cosmos-include-faint`: Use both bright and faint COSMOS catalogs (default: bright only)
- `--bin-data-from-cosmos`: Recompute noise/mask from COSMOS (else use precomputed `PATH_TO_STD_NOISE`, `PATH_TO_MASK`)

### Training Arguments (train.py specific)

- `-a`, `--arch`: Architecture ("SUNetNoiseAware" for denoisers, "UNetPreproc" for DeepMass)
- `-m`, `--mode`: "wiener" for preprocessing (DeepMass only)
- `-d`: Denoiser mode (default False)
- `-ng`: Non-Gaussian mode (residual networks, use with `-d` for denoisers)
- `-uq`: Enable uncertainty quantification (variance/order-2 training)
- `-t1 TIMESTAMP`: Point estimate checkpoint timestamp (required for order-2 training with `-uq`)
- `-e1 EPOCHS`: Number of epochs for point estimate model (required for order-2 training with `-uq`)
- `-b`, `--batch-size`: Training batch size
- `-e`, `--nepochs`: Number of epochs (for order-1 or order-2 network)
- `-lr`: Learning rate
- `--lr-scheduler`: Enable learning rate decay
- `-c`, `--config-name`: Custom model name suffix (for output paths)
- `--bin-data-from-cosmos`: Recompute noise/mask from COSMOS catalog (else use precomputed from config)
- `--cosmos-include-faint`: Use both bright and faint COSMOS catalogs when `--bin-data-from-cosmos` is set

### Inference Arguments (mass mapping scripts)

- `-c`, `--model-name`: Model config string (base name of checkpoint directory)
- `-a`, `--arch`: Architecture used in training
- `-t`, `--model-timestamp`: Checkpoint timestamp (YYYYMMDD_hhmmss)
- `-m`, `--mode`: Mode ("wiener" for preprocessing, "residual" for residual denoiser)
- `-alph`, `--step-sizes`: Step size values (0.0–1.0 range, relative to upper limit)
- `-i`, `--niter`: Number of iterations
- `-uq`: Enable uncertainty quantification (requires `-t0`)
- `-t0 TIMESTAMP`: Calibration model checkpoint timestamp (required for `-uq`)
- `--cqr`: Enable Conformal Quantile Regression
- `--find-optimal-hyperparam-precalib`: Auto-tune CQR hyperparameter on calibration set
- `--test-on-real-data`: Run inference on real COSMOS shear map
- `--run-both`: Run on both simulated test set and real COSMOS shear map
- `--save-tensors`: Save output tensors to disk
- `--nimgs-save`: Number of images to save when using `--save-tensors`
- `-o`: Custom output suffix
- `--bin-data-from-cosmos`: Recompute noise/mask from COSMOS (else use precomputed)
- `--cosmos-include-faint`: Use both bright and faint COSMOS catalogs

### Physics Parameters

- Opening angle, redshift bins, and lensing geometry configured in dataset classes
- Default opening angle: 15 arcmin (configurable per script)
- Convergence/shear conversions in `physics.py`

## Notebooks

Quick reference to example notebooks:

| Notebook | Purpose |
|----------|---------|
| `distribution_free_uq.ipynb` | Demonstrates distribution-free uncertainty quantification workflow |
| `pnpmass.ipynb` | PnPMass algorithm usage and results |
| `massmapping_uq.ipynb` | Mass mapping with uncertainty visualization |
| `get_std_noise_mask_gamma_from_cosmos.ipynb` | Computing noise stats and masks from COSMOS |
| `sandbox.ipynb` | Experimental code and exploration |

## Common Tasks

### Create a Dataset

```bash
# Training/validation set with data augmentation
python scripts/create_augmented_dataset.py \
  -o ~/data/train_val.hdf5 --idx-lp 2 \
  --angle-batch-size 8 --angle-step 1 --niter-per-angle 2 \
  -w 8 --seed 42 -v

# Optional: with redshift bins
python scripts/create_augmented_dataset.py \
  -o ~/data/train_val_redshift.hdf5 --idx-lp 2 -z \
  --angle-batch-size 8 --angle-step 1 --niter-per-angle 2 \
  -w 8 --seed 42 -v

# Calibration set
python scripts/create_augmented_dataset.py \
  -o ~/data/calib.hdf5 --idx-lp 1 \
  --angle-batch-size 5 --angle-step 8 \
  -w 5 --seed 42 -v

# Test set (center-cropped)
python scripts/create_cropped_dataset.py \
  -o ~/data/test.hdf5 --idx-lp 1 \
  --seed 42 -v
```

### Train Denoisers (PnPMass)

#### Point Estimate (Order-1)

**Standard version:**
```bash
python scripts/train.py \
  -a SUNetNoiseAware -d \
  --scale 0.2 --scale-min 0.0 \
  -b 16 -e 100 -lr 1e-3 --lr-scheduler \
  -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -w 8 --seed 42 -v
```

**Residual (non-Gaussian) version:**
```bash
python scripts/train.py \
  -a SUNetNoiseAware -d -ng \
  --scale 0.2 --scale-min 0.0 \
  -b 16 -e 100 -lr 1e-3 --lr-scheduler \
  -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -w 8 --seed 42 -v
```

#### Variance Estimate (Order-2)

**Standard version** (requires trained point estimate checkpoint):
```bash
python scripts/train.py \
  -a SUNetNoiseAware -d -uq \
  -t1 20250613_143319 -e1 100 \
  --scale 0.2 --scale-min 0.0 \
  -b 16 -e 100 -lr 1e-3 --lr-scheduler \
  -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -w 8 --seed 42 -v
```

**Residual version** (requires trained residual point estimate checkpoint):
```bash
python scripts/train.py \
  -a SUNetNoiseAware -d -ng -uq \
  -t1 20250716_170944 -e1 100 \
  --scale 0.2 --scale-min 0.0 \
  -b 16 -e 100 -lr 1e-3 --lr-scheduler \
  -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -w 8 --seed 42 -v
```

### Train DeepMass

#### Point Estimate (Order-1)

**COSMOS bright catalog only:**
```bash
python scripts/train.py \
  -a UNetPreproc -m wiener --bin-data-from-cosmos \
  -e 20 --lr-scheduler \
  -c deepmass_arch_UNetPreproc_mode-preproc_wiener_nepochs_20 \
  -w 8 --seed 42 -v
```

**COSMOS bright + faint catalogs:**
```bash
python scripts/train.py \
  -a UNetPreproc -m wiener --bin-data-from-cosmos --cosmos-include-faint \
  -e 20 --lr-scheduler \
  -c deepmass_arch_UNetPreproc_mode-preproc_wiener_brightfaint_nepochs_20 \
  -w 8 --seed 42 -v
```

#### Variance Estimate (Order-2)

**Note:** Order-2 trains for 100 epochs (vs 20 for order-1) due to validation loss still decreasing.

**COSMOS bright catalog only:**
```bash
python scripts/train.py \
  -a UNetPreproc -m wiener --bin-data-from-cosmos -uq \
  -t1 20250613_143319 -e1 20 \
  -e 100 --lr-scheduler \
  -c deepmass_arch_UNetPreproc_mode-preproc_wiener_nepochs_20 \
  -w 8 --seed 42 -v
```

**COSMOS bright + faint catalogs:**
```bash
python scripts/train.py \
  -a UNetPreproc -m wiener --bin-data-from-cosmos --cosmos-include-faint -uq \
  -t1 20250613_143319 -e1 20 \
  -e 100 --lr-scheduler \
  -c deepmass_arch_UNetPreproc_mode-preproc_wiener_brightfaint_nepochs_20 \
  -w 8 --seed 42 -v
```

### Run PnPMass Inference

#### Basic Inference (Test Step Sizes)

**Standard version:**
```bash
python scripts/pnpmass.py \
  -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250613_143319 \
  -alph 0.5 0.75 1.0 \
  -i 8 -w 8 \
  -o niter_8 --save-tensors --nimgs-save 8 \
  --seed 42 -v
```

**Residual version:**
```bash
python scripts/pnpmass.py \
  -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250716_170944 --mode residual \
  -alph 0.5 0.75 1.0 \
  -i 8 -w 8 \
  -o mode_residual_niter_8 --save-tensors --nimgs-save 8 \
  --seed 42 -v
```

#### With Uncertainty Quantification & Conformal Prediction

**Standard version:**
```bash
python scripts/pnpmass.py \
  -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250613_143319 \
  -uq -t0 20250903_164013 \
  -i 8 --cqr --find-optimal-hyperparam-precalib \
  -w 8 --save-tensors \
  -o niter_8_cqr \
  --seed 42 -v
```

**Residual version:**
```bash
python scripts/pnpmass.py \
  -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250716_170944 --mode residual \
  -uq -t0 20250903_164205 \
  -i 8 --cqr --find-optimal-hyperparam-precalib \
  -w 8 --save-tensors \
  -o mode_residual_niter_8_cqr \
  --seed 42 -v
```

#### On Real COSMOS Shear Map

**Standard version:**
```bash
python scripts/pnpmass.py \
  --bin-data-from-cosmos --test-on-real-data \
  -c denoiser_arch_SUNetNoiseAware_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250613_143319 \
  -uq -t0 20250903_164013 \
  -i 8 --cqr --find-optimal-hyperparam-precalib \
  -w 8 --save-tensors \
  -o cosmos_niter_8_cqr \
  --seed 42 -v
```

**Residual version:**
```bash
python scripts/pnpmass.py \
  --bin-data-from-cosmos --test-on-real-data \
  -c denoiser_arch_SUNetNoiseAware_nongaussian_scale_0.2_scale-min_0.0_batch-size_16_nepochs_100_learning-rate_1e-3 \
  -a SUNetNoiseAware \
  -t 20250716_170944 --mode residual \
  -uq -t0 20250903_164205 \
  -i 8 --cqr --find-optimal-hyperparam-precalib \
  -w 8 --save-tensors \
  -o cosmos_mode_residual_niter_8_cqr \
  --seed 42 -v
```

**Options:**
- Use `--run-both` instead of `--test-on-real-data` to run on both simulated test set and real shear map
- Add `--cosmos-include-faint` to use both bright and faint COSMOS catalogs
- Omit `--bin-data-from-cosmos` if using precomputed noise/mask from config (`PATH_TO_STD_NOISE`, `PATH_TO_MASK`)

### Other Mass Mapping Methods

Available scripts follow the same patterns as PnPMass:

- **DeepMass** (`scripts/deepmass.py`): Direct deep learning approach
- **MCALens** (`scripts/mcalens.py`): Iterative algorithm with maximum contrast
- **Iterative Wiener** (`scripts/wiener.py`): Classic iterative Wiener filtering
- **Kaiser-Squires** (`scripts/ks.py`): Direct analytical mass mapping

Consult individual scripts for method-specific arguments.

## Testing

No automated test framework is currently configured. Validation is done through:
- Jupyter notebook examples in `notebooks/`
- Custom experiment scripts in `scripts/sandbox.py`
- Visual inspection of inference results and uncertainty estimates

To verify installation:

```bash
python -c "import wlmmuq; print(wlmmuq.__file__); print(f'Config: {wlmmuq.config.CONFIGFILE}')"
```

## Important Implementation Notes

- **Lazy import**: `wlmmuq.config` searches for configuration at import time—modify config before importing
- **SLURM integration**: Job scripts in `slurm/` directory for cluster execution
- **Verbosity**: Most scripts accept `-v` flag for detailed logging
- **Seed management**: Use `--seed 42` for reproducibility across data augmentation and training
- **Batch size distinction**: Data scripts use `-b` for I/O batch (memory), training uses `-b` for gradient updates
- **Model checkpoints**: Stored with automatic naming; retrieve via `-t TIMESTAMP` flag
- **HDF5 datasets**: Use HDF5 format for all data files; check `datasets/base_dataset.py` for I/O patterns
