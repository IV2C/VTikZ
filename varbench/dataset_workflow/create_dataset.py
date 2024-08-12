"""
Creates a huggingface dataset from the folder "dataset"

"""

from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import os
import pygit2
import argparse
from .diffcompute import diffcompute

# login(token=os.environ.get("HF_TOKEN"))


### functions


### parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
parser.add_argument("--hf_token", type=str, required=True, help="token for huggingface")
args = parser.parse_args()

### Main parameters
init_commit_message = "initial commit"
update_commit_message = "force update repository"

### dataset creation code
dataset_dict = {}


dataset_path = args.dataset

diffcompute(dataset_path)

for split in os.listdir(dataset_path):
    current_split = []
    for entry in os.listdir(os.path.join(dataset_path, split)):
        current_split.append({})#TODO
    dataset_dict[split] = current_split

dataset = Dataset.from_dict(pd.DataFrame(dataset_dict))
dataset.push_to_hub("CharlyR/varbench")
