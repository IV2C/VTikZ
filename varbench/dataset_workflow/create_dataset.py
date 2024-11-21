"""
Creates a huggingface dataset from the folder "dataset"

"""

import PIL.Image
from datasets import Dataset, Features, Sequence, Value, Image
import PIL
from huggingface_hub import login
import pandas as pd
import os
import argparse

from varbench.renderers import Renderer, SvgRenderer, TexRenderer
from .patchcompute import patch_compute
import json

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
    match (subset):
        case "tikz":
            renderer: Renderer = TexRenderer()
        case "svg":
            renderer: Renderer = SvgRenderer()
    for entry in os.listdir(os.path.join(dataset_path, subset)):


        entry_path = os.path.join(dataset_path, subset, entry)
        #getting input code
        input_code = open(
            os.path.join(
                dataset_path,
                subset,
                entry,
                [
                    filename
                    for filename in os.listdir(entry_path)
                    if "input" in filename
                ][0],
            )
        ).read()

        #computing image input
        image_input = renderer.from_string_to_image(input_code)

        #getting the annotations of the current row 
        data = open(os.path.join(entry_path, "data.json")).read()
        data = json.loads(data)

        patches,solutions = patch_compute(entry_path)

        #Computing image solution
        solution_path = os.path.join(
            entry_path,
            "solutions",
            os.listdir(os.path.join(entry_path, "solutions"))[0],
        )
        with open(solution_path, "r") as solution_image_text:
            image_solution: PIL.Image.Image = renderer.from_string_to_image(
                solution_image_text.read()
            )
        


        current_subset.append(
            {
                "id": entry,
                "code": input_code,
                "instruction": data["instruction"],
                "result_description": data["result_description"],
                "difficulty":data["difficulty"],
                "patches": patches,
                "code_solutions":solutions,
                "image_solution": image_solution,
                "image_input":image_input
            }
        )
    if len(current_subset) > 0:
        dataset_dict[subset] = current_subset


features = Features(
    {
        "difficulty": Value("string"),
        "id": Value("string"),
        "code": Value("string"),
        "instruction": Value("string"),
        "result_description": Value("string"),
        "patches": Sequence(Value("string")),
        "code_solutions": Sequence(Value("string")),
        "image_solution": Image(),
        "image_input": Image()
    }
)

for subset in dataset_dict:
    current_subset = pd.DataFrame(dataset_dict[subset])
    dataset = Dataset.from_dict(pd.DataFrame(current_subset), features=features)
    dataset.push_to_hub("CharlyR/varbench", config_name=subset, split="test")
