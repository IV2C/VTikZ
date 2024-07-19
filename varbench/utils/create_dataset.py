"""creates a huggingface dataset from the folder "dataset"
"""
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import os


print(os.environ.get("HF_TOKEN"))
login(token=os.environ.get("HF_TOKEN"))

dataset_dict = {}

dataset_path = "dataset"

for split in os.listdir(dataset_path):
    dataset_dict[split] = {}
    for set in os.listdir(os.path.join(dataset_path, split)):
        input_content = open(os.path.join(dataset_path, split, set, "input")).read()
        reference_content = open(os.path.join(dataset_path, split, set, "reference")).read()
        prompt = open(os.path.join(dataset_path, split, set, "prompt")).read()
        dataset_dict[split][set] = {"input": input_content, "reference": reference_content, "prompt": prompt}


dataset = Dataset.from_dict(pd.DataFrame(dataset_dict))
dataset.push_to_hub("CharlyR/varbench")

