# Weak lensing mass mapping with uncertainty quantification

## Requirements and settings

#### Conda virtual environment

```bash
conda env create -f env.yml
conda activate wlmmuq
```

#### Additional packages

Some packages must be installed manually.

- `pycs` library, from the `cosmostat` repository (https://github.com/CosmoStat/cosmostat). Tested with commit nb `3eff4935bc3cd2368844c67452e429e0f4e7a127`. If `python -m pip install .` fails, simply specify the path to the git repository in `config.yml` (see below). Otherwise, leave it blank.

- `pysparse` Python bindings, from the `Sparse2D` repository (https://github.com/CosmoStat/Sparse2D). Tested with commit nb `3f9d54863765980299cfe92e0624ba93ed7ff02b`.

#### Configuration file

Update `config.yml` provided at the root of this repository, to configure data directories and file paths:

- `cosmos_dir`: Path to the COSMOS S10 weak lensing shear catalog (Schrabback et al. 2010). The directory contains data files named `cosmos_bright_cat_min.asc` and `cosmos_faint_cat.asc`.
- `ktng_dir`: Path to the $\kappa$TNG dataset of cosmological hydrodynamic simulations. See `https://github.com/0satoken/kappaTNG` to download the dataset. The directory contains HDF5 files named `LP001_run[001-100]_maps.hdf5`.
- `pycs_dir`: Path to the `pycs` library (see above). This should be used only if the `pip` installation is unsuccessful. Otherwise, leave it blank.
- `pickle_dir`: Path to the folder where the pickled objects will be stored. Used when running the script `massmapping.py`.

#### Note

If you encounter the error `ImportError: libpython3.11.so.1.0: cannot open shared object file: No such file or directory`, you can create a symbolic link to resolve it. Typically, the `libpython3.11.so.1.0` file is located in the `~/miniconda3/envs/wlmmuq/lib` directory within your virtual environment. You can link this file to a standard root location such as `/lib/x86_64-linux-gnu` by running the following command:
```sh
ln -s ~/miniconda3/envs/wlmmuq/lib/libpython3.11.so.1.0 /lib/x86_64-linux-gnu/libpython3.11.so.1.0
```
Please note that this workaround is not recommended as it makes some libraries available outside the virtual environment, which could lead to potential conflicts. However, this solution is provided here in the absence of a better alternative.

## Jupyter notebook reproducing our experiments

Check `wlmmuq.ipynb`.

## License

Copyright 2024 Hubert Leterme

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
