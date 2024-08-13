from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline
import os
import argparse
from transformers.pipelines.pt_utils import KeyDataset
from .evaluation.evaluator import evaluate_svg, evaluate_tikz,evaluation_dispatcher
# login(token=os.environ.get("HF_TOKEN"))




parser = argparse.ArgumentParser()
parser.add_argument("--subsets",nargs="+", type=str, required=True, help="Name of the subset(s) to evaluate the model on")
parser.add_argument("--models",nargs="+", type=str, required=True, help="Name of the model(s) to evaluate")
args = parser.parse_args()

subsets = args.subsets
models = args.models


#loading pipeline
for model in models:
    pipe = pipeline("text-generation",model=model)
    for subset in subsets:
        dataset = load_dataset("CharlyR/varbench", subset, split="train")
        evaluation_dispatcher[subset](dataset,pipe)





