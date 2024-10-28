from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline, GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from transformers.pipelines.pt_utils import KeyDataset
from .evaluation.evaluator import evaluate
from .model import LLM_Model,VLLM_model,API_model,ModelType

# login(token=os.environ.get("HF_TOKEN"))

    
def model_type_mapper(value):
    api_map = {
        'API':ModelType.API,
        'VLLM':ModelType.VLLM
    }
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
    default=["tikz","svg"]
)
parser.add_argument(
    "--model_type",
    "-t",
    type=model_type_mapper,
    required=True,
    help="type of the model to evaluate",
    default=ModelType.VLLM
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Name of the model to evaluate",
)

parser.add_argument(
    "--gpu_number",
    type=int,
    default=1,
    help="GPU number to use for evaluation"
)

parser.add_argument(
    "--api_url",
    type=str,
    default="https://api.openai.com/v1",
    help="URL of the openai completion compatible API"
)

parser.add_argument(
    "--api_key",
    type=str,
    default=None,
    help="API key for authentication, will default to the ENV variable OPENAI_API_KEY"
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature setting for model sampling"
)


args = parser.parse_args()

subsets = args.subsets
model_type = args.model_type

key_args:dict = {}
key_args["model_name"] = args.model
key_args["gpu_number"] = args.gpu_number
key_args["api_url"] = args.api_url
key_args["api_key"] = args.api_key
key_args["temperature"] = args.temperature


llm_model:LLM_Model = None
# loading model
match model_type:
    case ModelType.API:
        llm_model = API_model(**key_args)
    case ModelType.VLLM:
        llm_model = VLLM_model(**key_args)

for subset in subsets:
    dataset = load_dataset("CharlyR/varbench", subset, split="train")
    evaluate(dataset, llm_model)
