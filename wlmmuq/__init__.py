import os
import warnings
import yaml

# Look for configuration file
CONFIG_DIRLIST = [
    os.getcwd(), # Project-local directory
    os.path.join(os.path.expanduser("~"), ".config", __name__), # User-specific config directory
    os.path.join("/etc", __name__), # System-wide config directory
    os.path.dirname(__file__), # Package-default directory
]

CONFIGFILE = None
_iter_config_dirlist = iter(CONFIG_DIRLIST)
while CONFIGFILE is None:
    try:
        configdir = next(_iter_config_dirlist)
    except StopIteration:
        break
    else:
        test_path = os.path.join(configdir, "config.yml")
        if os.path.isfile(test_path):
            CONFIGFILE = test_path

if CONFIGFILE is None:
    warnings.warn("No configuration file provided.")
    CONFIG_DATA = {}
else:
    with open(CONFIGFILE, 'r', encoding='utf-8') as stream:
        CONFIG_DATA = yaml.safe_load(stream)
    if CONFIG_DATA['verbose']:
        print(f"Configuration file found in {configdir}")

COSMOS_DIR = CONFIG_DATA.get('cosmos_dir', None)
if COSMOS_DIR is not None:
    COSMOS_DIR = os.path.expanduser(COSMOS_DIR)
KTNG_DIR = CONFIG_DATA.get('ktng_dir', None)
if KTNG_DIR is not None:
    KTNG_DIR = os.path.expanduser(KTNG_DIR)
MODEL_DIR = CONFIG_DATA.get('model_dir', None)
if MODEL_DIR is not None:
    MODEL_DIR = os.path.expanduser(MODEL_DIR)
RESULTS_DIR = CONFIG_DATA.get('results_dir', None)
if RESULTS_DIR is not None:
    RESULTS_DIR = os.path.expanduser(RESULTS_DIR)

PATH_TO_STD_NOISE = CONFIG_DATA.get('path_to_std_noise', None)
if PATH_TO_STD_NOISE is not None:
    PATH_TO_STD_NOISE = os.path.expanduser(PATH_TO_STD_NOISE)
PATH_TO_MASK = CONFIG_DATA.get('path_to_mask', None)
if PATH_TO_MASK is not None:
    PATH_TO_MASK = os.path.expanduser(PATH_TO_MASK)
PATH_TO_EXTENT = CONFIG_DATA.get('path_to_extent', None)
if PATH_TO_EXTENT is not None:
    PATH_TO_EXTENT = os.path.expanduser(PATH_TO_EXTENT)
PATH_TO_PS = CONFIG_DATA.get('path_to_ps', None)
if PATH_TO_PS is not None:
    PATH_TO_PS = os.path.expanduser(PATH_TO_PS)
PATH_TO_REAL_SHEARMAP = CONFIG_DATA.get('path_to_real_shearmap', None)
if PATH_TO_REAL_SHEARMAP is not None:
    PATH_TO_REAL_SHEARMAP = os.path.expanduser(PATH_TO_REAL_SHEARMAP)
PATH_TO_XCLUS = CONFIG_DATA.get('path_to_xclus', None)
if PATH_TO_XCLUS is not None:
    PATH_TO_XCLUS = os.path.expanduser(PATH_TO_XCLUS)

PATH_TO_TRAIN_VAL_DATASET = CONFIG_DATA.get('path_to_train_val_dataset', None)
if PATH_TO_TRAIN_VAL_DATASET is not None:
    PATH_TO_TRAIN_VAL_DATASET = os.path.expanduser(PATH_TO_TRAIN_VAL_DATASET)
PATH_TO_TEST_DATASET = CONFIG_DATA.get('path_to_test_dataset', None)
if PATH_TO_TEST_DATASET is not None:
    PATH_TO_TEST_DATASET = os.path.expanduser(PATH_TO_TEST_DATASET)
PATH_TO_CALIB_DATASET = CONFIG_DATA.get('path_to_calib_dataset', None)
if PATH_TO_CALIB_DATASET is not None:
    PATH_TO_CALIB_DATASET = os.path.expanduser(PATH_TO_CALIB_DATASET)

TRAIN_VAL_DATASET_NAME = CONFIG_DATA.get('train_val_dataset_name', None)
TEST_DATASET_NAME = CONFIG_DATA.get('test_dataset_name', None)
REAL_SHEARMAP_NAME = CONFIG_DATA.get('real_shearmap_name', None)

LEARNLETS_PRETRAINED_WEIGHTS_DIR = CONFIG_DATA.get('learnlets_pretrained_weights_dir', None)
if LEARNLETS_PRETRAINED_WEIGHTS_DIR is not None:
    LEARNLETS_PRETRAINED_WEIGHTS_DIR = os.path.expanduser(LEARNLETS_PRETRAINED_WEIGHTS_DIR)

KEY_REPLACEMENT_DICT = CONFIG_DATA.get("key_replacement_dict", None)
