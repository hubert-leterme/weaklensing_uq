import os
import sys
import yaml
import warnings

# First, check if the config file wcnn_config.yml is in the current directory.
# If not in there, check in "~/.config/".
CONFIG_DIRLIST = [
    os.getcwd(), # run a script from the directory where the config file is located
    os.path.dirname(os.getcwd()), # run a notebook in the 'notebook/' directory
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
        test_path = os.path.join(configdir, "wlconfig.yml")
        if os.path.isfile(test_path):
            CONFIGFILE = test_path

if CONFIGFILE is None:
    warnings.warn("No configuration file provided.")
    CONFIG_DATA = None
else:
    with open(CONFIGFILE, 'r', encoding='utf-8') as stream:
        CONFIG_DATA = yaml.safe_load(stream)
    if CONFIG_DATA['verbose']:
        print(f"Configuration file found in {configdir}")
