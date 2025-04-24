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

### dataset creation code
dataset_dict = {}

dataset_path = args.dataset


for subset in os.listdir(dataset_path):
    current_subset = []
    match (subset):
        case "tikz":
            renderer: Renderer = TexRenderer(debug=True)
        case "svg":
            renderer: Renderer = SvgRenderer()
    for entry in sorted(os.listdir(os.path.join(dataset_path, subset))):
        entry_path = os.path.join(dataset_path, subset, entry)
        logger.info(f"adding {entry_path}")
        # getting input code
        input_path = os.path.join(
            entry_path,
            [filename for filename in os.listdir(entry_path) if "input" in filename][0],
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

        current_subset.append(
            {
                "difficulty_ast": ted,
                "id": entry,
                "code": unified_input_code,
                "commented_code": commented_input_code,
                "instruction": data["instruction"],
                "result_description": data["result_description"],
                "difficulty": data["difficulty"],
                "modification_type": data["modif_type"],
                "type": data["type"],
                "patch": patches,
                "template_solution_code": uncommented_template_solution_codes,
                "code_solution": unified_solution_codes,
                "image_solution": images_solution,
                "image_input": image_input,
            }
        )
    if len(current_subset) > 0:
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

for subset in dataset_dict:
    current_subset = pd.DataFrame(dataset_dict[subset])
    dataset = Dataset.from_dict(pd.DataFrame(current_subset), features=features)
    #dataset = dataset.filter(lambda row: row["id"] == "bee_three_wings" or row["id"] == "bee_red_stripes")#temporary for test
    dataset.push_to_hub(
        "CharlyR/vtikz", config_name=subset, split="benchmark")

    dataset_test = dataset.filter(lambda row: row["difficulty"] == "medium").select(
       [6, 7]
    )
    dataset_test.push_to_hub("CharlyR/vtikz", config_name=subset, split="test")
