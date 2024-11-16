from datasets import load_dataset
import os
import argparse

from varbench.api.chat_api import ChatApi
from varbench.renderers import Renderer, SvgRenderer, TexRenderer
from varbench.utils.model_launch import launch_model
from .evaluation.evaluator import evaluate
from .agent import SimpleLLMAgent
import json

from loguru import logger

# login(token=os.environ.get("HF_TOKEN"))

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
    "--run-model",
    "-r",
    action="store_true",
    help="Name ",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Name of the model to evaluate",
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

args = parser.parse_args()

subsets = args.subsets

key_args: dict = {}
key_args["model_name"] = args.model
key_args["api_url"] = args.api_url
key_args["api_key"] = args.api_key
key_args["temperature"] = args.temperature
key_args["run_model"] = args.run_model
key_args["n"] = args.passk


# loading model
if key_args["run_model"]:
    if key_args["api_url"]:
        logger.warning("found run-model and api_url parameters, api_url will be ignored")
    key_args["api_url"] = launch_model(key_args["model_name"])
    
if not key_args["api_key"]:
    key_args["api_key"] = os.environ.get("OPENAI_API_KEY")

#instantiating api
api:ChatApi = ChatApi.from_url(**key_args)


agent = SimpleLLMAgent(api)


if not os.path.exists("./results"):
    os.mkdir("./results")

result_path = os.path.join("./results", args.model.replace("/", "_"))

if not os.path.exists(result_path):
    os.mkdir(result_path)


for subset in subsets:
    dataset = load_dataset("CharlyR/varbench", subset, split="test")

    # creating compiler
    match subset:
        case "tikz":
            renderer = TexRenderer()
        case "svg":
            renderer = SvgRenderer()
        case _:
            logger.warning("unsupported subset "+ subset+", skipping")
            continue

    # evaluating
    result_scores, score_dataset = evaluate(dataset, agent, renderer)
    logger.info(result_scores)
    score_dataset.to_csv(os.path.join(result_path, subset + ".csv"))
    with open(os.path.join(result_path, subset + ".json"), "w") as subset_result:
        subset_result.write(json.dumps(result_scores))
