"""
Creates a huggingface dataset from the folder "dataset"

"""

from collections import defaultdict
import PIL.Image
from datasets import Dataset, Features, Sequence, Value, Image
import PIL
from huggingface_hub import login
import pandas as pd
import os
import argparse
import re
from .ast_difficulty_compute import TED_tikz
from vtikz.renderers import Renderer, SvgRenderer, TexRenderer
import json
from loguru import logger
from .utils import uncomment_code, unify_code, patch_compute, create_default

login(token=os.environ.get("HF_TOKEN"))


### parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="path to the dataset")
args = parser.parse_args()


# list the tags
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
ds_inf = api.list_repo_refs("CharlyR/VTikz", repo_type="dataset")
print("Existing tags:")
print(" | ".join([tag.name for tag in ds_inf.tags]))
### ask the tag to the user
new_tag = input("New dataset tag: ")

### dataset creation code
dataset_dict: dict[str, dict] = {}

dataset_path = args.dataset


for subset in [
    folder for folder in os.listdir(dataset_path) if not folder.startswith(".")
]:
    current_subset: dict[str, dict[str,list]] = {}
    match (subset):
        case "tikz":
            renderer: Renderer = TexRenderer(debug=True)
        case "svg":
            renderer: Renderer = SvgRenderer()
    for split_name in ["benchmark", "test"]:
        logger.info(f"Computing the {subset} config, and split {split_name}.")

        if not os.path.exists(os.path.join(dataset_path, subset, split_name)):
            continue
        current_config_subset = defaultdict(list)
        for entry in sorted(os.listdir(os.path.join(dataset_path, subset, split_name))):
            entry_path = os.path.join(dataset_path, subset, split_name, entry)
            logger.info(f"adding {entry_path}")
            # getting input code
            input_path = os.path.join(
                entry_path,
                [
                    filename
                    for filename in os.listdir(entry_path)
                    if "input" in filename
                ][0],
            )
            commented_input_code = open(input_path).read()
            unified_input_code = uncomment_code(commented_input_code)
            # getting solution codes
            solution_folder = os.path.join(entry_path, "solutions")
            solution_paths = [
                os.path.join(solution_folder, sol_name)
                for sol_name in os.listdir(solution_folder)
            ]

            commented_solution_codes = [
                open(sol_path).read() for sol_path in solution_paths
            ]
            uncommented_template_solution_codes = [
                uncomment_code(commented_solution_code)
                for commented_solution_code in commented_solution_codes
            ]  # uncommenting
            unified_solution_codes = [
                create_default(uncommented_template_solution_code)
                for uncommented_template_solution_code in uncommented_template_solution_codes
            ]  # creating the default implementations

            # Computing the patches
            patches = [
                patch_compute(unified_input_code, unified_solution_code)
                for unified_solution_code in unified_solution_codes
            ]

            # computing image input
            image_input = renderer.from_string_to_image(unified_input_code)
            image_input = image_input.resize((300, 300))  # TODO make a parameter

            # getting the annotations of the current row
            data = open(os.path.join(entry_path, "data.json")).read()
            data = json.loads(data)

            # Computing images solution
            images_solution: list[PIL.Image.Image] = [
                renderer.from_string_to_image(unified_solution_code)
                for unified_solution_code in unified_solution_codes
            ]
            images_solution = [
                image_solution.resize((300, 300)) for image_solution in images_solution
            ]  # TODO make parameter
            ted: list[int] = [
                TED_tikz(unified_input_code, unified_solution_code)
                for unified_solution_code in unified_solution_codes
            ]

            current_config_subset["difficulty_ast"].append(ted)
            current_config_subset["id"].append(entry)
            current_config_subset["code"].append(unified_input_code)
            current_config_subset["commented_code"].append(commented_input_code)
            current_config_subset["instruction"].append(data["instruction"])
            current_config_subset["result_description"].append(data["result_description"])
            current_config_subset["difficulty"].append(data["difficulty"])
            current_config_subset["modification_type"].append(data["modif_type"])
            current_config_subset["type"].append(data["type"])
            current_config_subset["patch"].append(patches)
            current_config_subset["template_solution_code"].append(uncommented_template_solution_codes)
            current_config_subset["code_solution"].append(unified_solution_codes)
            current_config_subset["image_solution"].append(images_solution)
            current_config_subset["image_input"].append(image_input)
            
            
        if len(current_config_subset) > 0:
            current_subset[split_name] = current_config_subset
    dataset_dict[subset] = current_subset


features = Features(
    {
        "difficulty_ast": Sequence(Value("float")),
        "difficulty": Value("string"),
        "id": Value("string"),
        "code": Value("string"),
        "commented_code": Value("string"),
        "template_solution_code": Sequence(Value("string")),
        "instruction": Value("string"),
        "result_description": Value("string"),
        "patch": Sequence(Value("string")),
        "modification_type": Value("string"),
        "type": Value("string"),
        "code_solution": Sequence(Value("string")),
        "image_solution": Sequence(Image()),
        "image_input": Image(),
    }
)

import pickle
with open("dataset/.cache/ds_pickle","wb") as dsp:
    pickle.dump(dataset_dict,dsp)

api.create_tag("CharlyR/vtikz", tag=new_tag,repo_type="dataset")

for config_name in dataset_dict:
    for split_name, subset in dataset_dict[config_name].items():
        ds = Dataset.from_dict(subset, features=features)
        ds.save_to_disk(f"dataset/.cache/{config_name}{split_name}")  # debug
        ds.push_to_hub("CharlyR/vtikz", config_name=subset, split=split_name)


#Does not work for some reason, but it works when using this script once this one is launched [dataset/.cache/dataset_test.py]