import argparse
import subprocess
import configparser
import os
import logging

#parser arg for the name of the entry

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str)

args = parser.parse_args()

entry_name = args.name

#find the corresponding path
submodules_config = configparser.ConfigParser()
submodules_config.read(".gitmodules")
submodule_path = submodules_config[f'submodule "{entry_name}"']["path"]

#deinit submodule

subprocess.run(["git", "submodule", "deinit","-f", submodule_path])

#remove in .git/modules
subprocess.run([ "rm", "-rf", ".git/modules/" + entry_name])

#remove from .gitmodules

subprocess.run(["git", "config","-f",".gitmodules","--remove-section","submodule." + entry_name])

# stage changes

subprocess.run(["git","add",".gitmodules"])

#remove from the cache

subprocess.run(["git", "rm","-r", "--cached", submodule_path])

#remove the entry from the dataset completely
submodule_parent = os.path.dirname(submodule_path)
subprocess.run(["rm", "-r", submodule_parent])





