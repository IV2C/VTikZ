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

from .ast_difficulty_compute import TED_tikz
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
        # getting input code
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

        # computing image input
        image_input = renderer.from_string_to_image(input_code)
        image_input = image_input.resize((300, 300))  # TODO make a parameter

        # getting the annotations of the current row
        data = open(os.path.join(entry_path, "data.json")).read()
        data = json.loads(data)

        patch, solution = patch_compute(entry_path)

        # Computing image solution
        solution_path = os.path.join(
            entry_path,
            "solutions",
            os.listdir(os.path.join(entry_path, "solutions"))[0],
        )
        with open(solution_path, "r") as solution_image_text:
            str_solution_code = solution_image_text.read()
            image_solution: PIL.Image.Image = renderer.from_string_to_image(
                str_solution_code
            )
            image_solution = image_solution.resize((300, 300))  # TODO make parameter
            ted = TED_tikz(input_code, str_solution_code)

        current_subset.append(
            {
                "difficulty_ast": ted,
                "id": entry,
                "code": input_code,
                "instruction": data["instruction"],
                "result_description": data["result_description"],
                "difficulty": data["difficulty"],
                "modification_type": data["modif_type"],
                "type": data["type"],
                "patch": patch,
                "code_solution": solution,
                "image_solution": image_solution,
                "image_input": image_input,
            }
        )
    if len(current_subset) > 0:
        dataset_dict[subset] = current_subset


features = Features(
    {
        "difficulty_ast": Value("float"),
        "difficulty": Value("string"),
        "id": Value("string"),
        "code": Value("string"),
        "instruction": Value("string"),
        "result_description": Value("string"),
        "patch": Value("string"),
        "modification_type": Value("string"),
        "type": Value("string"),
        "code_solution": Value("string"),
        "image_solution": Image(),
        "image_input": Image(),
    }
)

for subset in dataset_dict:
    current_subset = pd.DataFrame(dataset_dict[subset])
    dataset = Dataset.from_dict(pd.DataFrame(current_subset), features=features)
    dataset.push_to_hub("CharlyR/varbench", config_name=subset, split="benchmark")

    dataset_test = dataset.filter(lambda row: row["difficulty"] == "medium").select(
        [6, 7]
    )
    dataset_test.push_to_hub("CharlyR/varbench", config_name=subset, split="test")
