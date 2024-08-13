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
import subprocess

login(token=os.environ.get("HF_TOKEN"))


### functions
def commit_changes(repo_path) -> str:
    """commits the changes in the repo

    Args:
        repo_path (str): The path to the repository

    Returns:
        str: the last commit hash
    """
    subprocess.run(["git", "add", "-A"], cwd=repo_path, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["git", "commit", "-m", "update entry for dataset creation"],
        cwd=repo_path,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["git", "push"], cwd=repo_path, stderr=subprocess.DEVNULL)
    git_cmd_result = subprocess.run(
        ["git", "log", "--format=%H", "-n", "1"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return git_cmd_result.stdout.replace("\n", "")


### parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
parser.add_argument("--hf_token", type=str, help="token for huggingface")#TODO
args = parser.parse_args()

### dataset creation code
dataset_dict = {}


dataset_path = args.dataset

diffcompute(dataset_path)

for split in os.listdir(dataset_path):
    current_split = []
    for entry in os.listdir(os.path.join(dataset_path, split)):
        commit_id = commit_changes(os.path.join(dataset_path, split, entry,"input"))
        current_split.append({
            "repository": "https://github.com/VarBench-SE/" + split,
            "commit_id": commit_id,
            "prompt": open(os.path.join(dataset_path, split, entry, "prompt.txt")).read(),
            "patch": open(os.path.join(dataset_path, split, entry, f"{entry}.patch")).read(),
            })
    dataset_dict[split] = current_split

dataset = Dataset.from_dict(pd.DataFrame(dataset_dict))
dataset.push_to_hub("CharlyR/varbench")
