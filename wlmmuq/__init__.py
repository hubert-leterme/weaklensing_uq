import os
import warnings
import yaml

# First, check if the config file wcnn_config.yml is in the current directory.
# If not in there, check in "~/.config/".
CONFIG_DIRLIST = [
    os.getcwd(), # run a script from the directory where the config file is located
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )), # look into the parent directory
    os.path.expanduser("~/.config") # or look into a generic directory
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

USE_PYCS = CONFIG_DATA.get('use_pycs', False)

COSMOS_DIR = os.path.expanduser(CONFIG_DATA.get('cosmos_dir', None))
KTNG_DIR = os.path.expanduser(CONFIG_DATA.get('ktng_dir', None))

PATH_TO_STD_NOISE = os.path.expanduser(CONFIG_DATA.get('path_to_std_noise', None))
PATH_TO_MASK = os.path.expanduser(CONFIG_DATA.get('path_to_mask', None))
PATH_TO_PS = os.path.expanduser(CONFIG_DATA.get('path_to_ps', None))

LEARNLETS_PRETRAINED_WEIGHTS_DIR = os.path.expanduser(
    CONFIG_DATA.get('learnlets_pretrained_weights_dir', None)
)
