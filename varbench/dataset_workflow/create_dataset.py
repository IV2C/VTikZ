"""
Creates a huggingface dataset from the folder "dataset"

"""

from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import login
import pandas as pd
import os
import argparse
from .diffcompute import diffcompute

login(token=os.environ.get("HF_TOKEN"))


### parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
args = parser.parse_args()

### dataset creation code
dataset_dict = {}

dataset_path = args.dataset


for subset in os.listdir(dataset_path):
    current_subset = []
    for entry in os.listdir(os.path.join(dataset_path, subset)):

        entry_path = os.path.join(dataset_path,subset,entry)
        input_code = open(
            os.path.join(
                dataset_path,
                subset,
                entry,
                [filename for filename in os.listdir(entry_path) if "input" in filename][0],
            )
        ).read()

        instruction = open(
            os.path.join(entry_path, "instruction.txt")
        ).read()

        diffs = diffcompute(entry_path)
        current_subset.append(
            {
                "id": entry,
                "code": input_code,
                "instruction": instruction,
                "diffs": diffs,
            }
        )
    if len(current_subset) > 0:
        dataset_dict[subset] = current_subset


print(dataset_dict)

features = Features(
    {
        "id": Value("string"),
        "code": Value("string"),
        "instruction": Value("string"),
        "diffs": Sequence(Value("string")),
    }
)

for subset in dataset_dict:
    current_subset = pd.DataFrame(dataset_dict[subset])
    dataset = Dataset.from_dict(pd.DataFrame(current_subset), features=features)
    dataset.push_to_hub("CharlyR/varbench", config_name=subset)
