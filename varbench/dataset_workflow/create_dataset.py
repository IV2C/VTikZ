"""
Creates a huggingface dataset from the folder "dataset"

"""
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import os
import pygit2
import argparse

#login(token=os.environ.get("HF_TOKEN"))




### functions

def create_repository(path:str,authorCommitter:str,initial_commit_message:str, git_token:str):
    """creates a git repository in the path passed as argument

    Args:
        path (str): the path where the repository will be created
        authorCommitter (str): the author and committer of the repository
        initial_commit_message (str): the initial commit message
        git_token (str): the token of the github account
    """
    repo = pygit2.init_repository(os.path.join(dataset_path, split, set, "input"))
    index = repo.index
    index.add(".")
    index.write()
    
    tree = index.write_tree()
    commit = repo.create_commit("refs/heads/main", authorCommitter, authorCommitter,initial_commit_message, tree)
    
    repo_name = path.split("/")[-1]
    remote_url = f"https://github.com/VarBench-SE/{repo_name}.git"
    remote = repo.remotes.create('origin', remote_url)

    credentials = pygit2.UserPass('x-access-token', git_token) 
    callbacks = pygit2.RemoteCallbacks(credentials=credentials)
    remote.push(['refs/heads/master'], callbacks=callbacks)
    
def update_repository(path:str,committer:str,update_commit_message:str, git_token:str):    
    """TODO
    """
    ...
    
### parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--hf_token", type=str, required=True)
parser.add_argument("--git_token", type=str, required=True)
parser.add_argument("--mail", type=str, required=True)
args = parser.parse_args()

### Main parameters
authorCommitter = pygit2.Signature("CharlyR", parser.mail)
committer = "CharlyR"
init_commit_message = "initial commit"
update_commit_message = "force update repository"

### dataset creation code
dataset_dict = {}

dataset_path = "dataset"

for split in os.listdir(dataset_path):
    dataset_dict[split] = {}
    for set in os.listdir(os.path.join(dataset_path, split)):
        if not os.path.exists(os.path.join(dataset_path, split, set, "input",".git")):
            create_repository(os.path.join(dataset_path, split, set),authorCommitter,init_commit_message, args.git_token)
dataset = Dataset.from_dict(pd.DataFrame(dataset_dict))
dataset.push_to_hub("CharlyR/varbench")

