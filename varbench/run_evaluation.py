from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from transformers.pipelines.pt_utils import KeyDataset

from varbench.renderers import Renderer, SvgRenderer, TexRenderer
from .evaluation.evaluator import evaluate
from .model import LLM_Model, VLLM_model, API_model, ModelType
import json

from loguru import logger

# login(token=os.environ.get("HF_TOKEN"))


def model_type_mapper(value):
    api_map = {"API": ModelType.API, "VLLM": ModelType.VLLM}
    if value.upper() not in api_map:
        raise argparse.ArgumentTypeError(f"Invalid API type: {value}")
    return api_map[value.upper()]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--subsets",
    "-s",
    nargs="+",
    type=str,
    help="Name of the subset(s) to evaluate the model on",
    default=["tikz", "svg"],
)
parser.add_argument(
    "--model_type",
    "-t",
    type=model_type_mapper,
    required=True,
    help="type of the model to evaluate",
    default=ModelType.VLLM,
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Name of the model to evaluate",
)

parser.add_argument(
    "--gpu_number", type=int, default=1, help="GPU number to use for evaluation"
)

parser.add_argument(
    "--api_url",
    type=str,
    default="https://api.openai.com/v1",
    help="URL of the openai completion compatible API",
)

parser.add_argument(
    "--api_key",
    type=str,
    default=None,
    help="API key for authentication, will default to the ENV variable OPENAI_API_KEY",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature setting for model sampling",
)
parser.add_argument(
    "--passk", type=int, default=1, help="Number of generations per prompt"
)
parser.add_argument(
    "--no_batch",
    default=False,
    action="store_true",
    help="whether or not to use batch requests, relevant when the api provider does not provide an equivalent",
)

args = parser.parse_args()

subsets = args.subsets
model_type = args.model_type

key_args: dict = {}
key_args["model_name"] = args.model
key_args["gpu_number"] = args.gpu_number
key_args["api_url"] = args.api_url
key_args["api_key"] = args.api_key
key_args["temperature"] = args.temperature
key_args["no_batch"] = args.no_batch
key_args["n"] = args.passk


llm_model: LLM_Model = None
# loading model
match model_type:
    case ModelType.API:
        llm_model = API_model(**key_args)
    case ModelType.VLLM:
        llm_model = VLLM_model(**key_args)

if not os.path.exists("./results"):
    os.mkdir("./results")

result_path = os.path.join("./results", args.model.replace("/", "_"))

if not os.path.exists(result_path):
    os.mkdir(result_path)


for subset in subsets:
    dataset = load_dataset("CharlyR/varbench", subset, split="train")

    # creating compiler
    renderer: Renderer = None
    match subset:
        case "tikz":
            renderer = TexRenderer()
        case "svg":
            renderer = SvgRenderer()

    # evaluating
    result_scores, score_dataset = evaluate(dataset, llm_model, renderer)
    logger.info(result_scores)
    score_dataset.to_csv(os.path.join(result_path, subset+".csv"))
    with open(os.path.join(result_path, subset+".json"), "w") as subset_result:
        subset_result.write(json.dumps(result_scores))
