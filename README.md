# Distribution-free uncertainty quantification for weak lensing mass mapping

## Requirements and settings

#### Conda virtual environment

For reproducibility.

```bash
conda env create -f env.yml
conda activate wlmmuq
```

#### Installation

Install `wlmmuq` library, provided in this repository, with `pip install .`.

#### Additional packages

To run Wiener and MCALens algorithms, the following packages must be installed manually.

- `pycs` library, from the `cosmostat` repository (https://github.com/CosmoStat/cosmostat). Tested with commit nb `3eff4935bc3cd2368844c67452e429e0f4e7a127`. If `python -m pip install .` fails, simply specify the path to the git repository in `config.yml` (see below). Otherwise, leave it blank.

- `pysparse` Python bindings, from the `Sparse2D` repository (https://github.com/CosmoStat/Sparse2D). Only needed for MCALens. Tested with commit nb `3f9d54863765980299cfe92e0624ba93ed7ff02b`.

#### Configuration file

Update `config.yml` provided at the root of this repository, to configure data directories and file paths:

- `use_tensorflow`: Boolean flag to indicate whether to import TensorFlow.
- `cosmos_dir`: Path to the COSMOS S10 weak lensing shear catalog (Schrabback et al. 2010). The directory contains data files named `cosmos_bright_cat_min.asc` and `cosmos_faint_cat.asc`.
- `ktng_dir`: Path to the $\kappa$TNG dataset of cosmological hydrodynamic simulations. See `https://github.com/0satoken/kappaTNG` to download the dataset. The directory contains a file named `zs.dat` as well as HDF5 files named `LP[XXX]/LP[XXX]_run[001-100]_maps.hdf5`, where `[XXX]` ranges from `001` to `100`.

#### Note

If you encounter the error `ImportError: libpython3.11.so.1.0: cannot open shared object file: No such file or directory` when working within a virtual environment, you may need to append the path to the directory for the shared libraries to `$LD_LIBRARY_PATH`. For example: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.

## Python scripts

**TODO: update.**

### Mass mapping with uncertainty quantification

#### Compute point estimates

```bash
python scripts/massmapping.py wiener wiener_niter_12_nimgs_225_${datetime} /path/to/test/dataset.hdf5 --niter 12 --nimgs 225 -b 45 --seed 42 -v
```

```bash
python scripts/massmapping.py mcalens mcalens_niter_${niter_mcalens}_Nsigma_5_nimgs_225_${datetime} /path/to/test/dataset.hdf5 --niter $niter_mcalens --Nsigma 5 --nimgs 225 -b 45 --seed 42 -v
```

### Creating an augmented dataset

Data augmentation by rotating and randomly cropping convergence maps. Used for training DeepMass.

```bash
python create_augmented_dataset.py path/to/destination/file.hdf5 --idx-lp 2 --nimgs 100 -b 25 --angle-batch-size 36 --angle-step 1 --niter-per-angle 2 --seed 42 -v
```

### Training DeepMass

```bash
python train.py path/to/augmented/dataset.hdf5 --input-method wiener --checkpoint-dir path/to/checkpoint --save-freq 8 --backup-dir path/to/backup -log path/to/log.csv --seed 42 -v
```

## Jupyter notebooks

Examples are provided in the Jupyter notebooks provided in the directory `./notebooks`.

## License

Copyright 2025 Hubert Leterme

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
