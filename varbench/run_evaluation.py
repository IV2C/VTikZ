from huggingface_hub import login
from datasets import load_dataset
import outlines
from transformers import pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from transformers.pipelines.pt_utils import KeyDataset
from .evaluation.evaluator import evaluation_dispatcher


# login(token=os.environ.get("HF_TOKEN"))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--subsets",
    "-s",
    nargs="+",
    type=str,
    required=True,
    help="Name of the subset(s) to evaluate the model on",
)
parser.add_argument(
    "--models",
    "-m",
    nargs="+",
    type=str,
    required=True,
    help="Name of the model(s) to evaluate",
)
parser.add_argument(
    "--instruction_tuned",
    "-it",
    action="store_true",
    help="If the model is an instruction tuned model",
    default=False,
)
args = parser.parse_args()

subsets = args.subsets
models = args.models
it_tuned = args.instruction_tuned


# loading pipeline
for model_name in models:
    generation_config = {"max_tokens": 1000}
    model = outlines.models.transformers(model_name)
    generator = outlines.generate.regex(
        model, "```(?:\n|\r\n)?([\s\S]*?)(?:\n|\r\n)?```"
    )
    for subset in subsets:
        dataset = load_dataset("CharlyR/varbench", subset, split="train")
        evaluation_dispatcher[subset](dataset, generator, it_tuned, generation_config)
