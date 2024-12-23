from datasets import load_dataset
import os
import argparse

from varbench.api.chat_api import ChatApi
from varbench.evaluation.metrics import (
    instantiate_agnostic_metrics,
    instantiate_non_agnostic_metrics,
)
from varbench.renderers import Renderer, SvgRenderer, TexRenderer
from varbench.utils.model_launch import launch_model
from varbench.evaluation.evaluator import evaluate, generate
from varbench.agents import instantiate_agent
import json

from datasets import Dataset

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
    "--metrics",
    "-me",
    nargs="+",
    type=str,
    help="Name of the metric(s) to evaluate on the output of the model",
    default=[
        "patch",
        "line",
        "clipImage",
        "clipText",
        "bleu",
        "bleuPatch",
        "crystalBleu",
        "crystalBleuPatch",
        "chrf",
        "chrfPatch",
        "TER",
        "TERPatch",
        "featureMatch",
        "LPIPS",
        "psnr",
        "msssim",
        "imageDiff",
        "MSE",
    ],
)


parser.add_argument(
    "--agent",
    "-a",
    type=str,
    help="Name of the agent to use",
    choices=["simpleLLM", "simpleLMM", "loopVLMLLM", "loopLMM"],
    default="simpleLLM",
    required=True,
)
### vlm and llm/vlm loop settings ##
parser.add_argument(
    "--vlm",
    "-v",
    type=str,
    help="vlm to use in the case of a loopVLMLLM",
    default=None,
)
parser.add_argument(
    "--vlm_api_url",
    type=str,
    help="URL of the openai completion compatible API",
)

parser.add_argument(
    "--vlm_api_key",
    type=str,
    default=None,
    help="API key for authentication, will default to the ENV variable OPENAI_API_KEY",
)
parser.add_argument(
    "--vlm-temperature",
    type=float,
    default=0,
    help="Temperature setting for vlm model sampling",
)
parser.add_argument(
    "--interaction-amount",
    type=int,
    default=2,
    help="number of interactions between the llm and the vlm",
)
#################
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
model_process = None
if key_args["run_model"]:
    if key_args["api_url"] and key_args["api_url"] != "":
        logger.warning(
            "found run-model and api_url parameters, api_url will be ignored"
        )
    key_args["api_url"], model_process = launch_model(key_args["model_name"])


if not key_args["api_key"]:
    key_args["api_key"] = os.environ.get("OPENAI_API_KEY")

# instantiating api
api: ChatApi = ChatApi.from_url(**key_args)

# Instantiating vlm api(if provided)
vlm_model: str = args.vlm
vlm_api_url: str = args.vlm_api_url
vlm_api_key: str = args.vlm_api_key
vlm_temperature: float = args.vlm_temperature
if vlm_model and vlm_model != "":
    vlm_api: ChatApi = ChatApi.from_url(
        vlm_temperature, 1, vlm_model, vlm_api_url, vlm_api_key
    )
else:
    vlm_api = None
interaction_amount: int = args.interaction_amount

# result path creation
if not os.path.exists("./results"):
    os.mkdir("./results")
split_used = "benchmark"
full_config_name = (
    args.agent
    + "_"
    + split_used
    + "_"
    + key_args["model_name"].replace("/", "_").replace("-", "")
    + "_t_"
    + str(key_args["temperature"])
    + (
        (
            "_"
            + vlm_model.replace("/", "_").replace("-", "")
            + "_t_"
            + str(vlm_temperature)
        )
        if vlm_model
        else ""
    )
)
# result path handling
result_path = os.path.join("./results", full_config_name)
generation_result_path = os.path.join(result_path, "generation")
evaluation_result_path = os.path.join(result_path, "evaluation")

if not os.path.exists(result_path):
    os.mkdir(result_path)
if not os.path.exists(generation_result_path):
    os.mkdir(generation_result_path)
if not os.path.exists(evaluation_result_path):
    os.mkdir(evaluation_result_path)


logger.info(
    f"Varbench Evaluation : split = {split_used} | agent = {args.agent} | model = {key_args["model_name"]} | api = {key_args["api_url"]}"
)
# generation
for subset in subsets:
    subset_generation_result_path = os.path.join(generation_result_path, subset)

    # skipping if exists
    if os.path.exists(subset_generation_result_path):
        logger.warning("Generated subset exists, skipping")
        continue

    logger.info(f"Starting generation on subset {str(subset)}")
    dataset = load_dataset("CharlyR/varbench", subset, split=split_used)

    # creating compiler
    match subset:
        case "tikz":
            renderer = TexRenderer()
        case "svg":
            renderer = SvgRenderer()
        case _:
            logger.warning("unsupported subset " + subset + ", skipping")
            continue

    # instantiate the agent
    agent = instantiate_agent(args.agent, api, vlm_api, renderer, interaction_amount)

    # generating and saving the dataset
    subset_processed: Dataset = generate(dataset, agent, renderer)
    subset_processed.save_to_disk(subset_generation_result_path)

# instantiating dataset agnostic metrics
agnostic_metrics = instantiate_agnostic_metrics(args.metrics)
# stopping the model if running
if model_process:
    model_process.terminate()
    model_process.kill()

# evaluation
for subset in subsets:

    logger.info(f"Starting evaluation on subset {str(subset)}")

    # loading existing dataset
    subset_generation_result_path = os.path.join(generation_result_path, subset)
    dataset = Dataset.load_from_disk(subset_generation_result_path)

    # instantiating non agnostic metrics
    metrics = agnostic_metrics + instantiate_non_agnostic_metrics(args.metrics)

    # evaluating
    score_dataset = evaluate(dataset, metrics)

    subset_evaluation_result_path = os.path.join(evaluation_result_path, subset)

    score_dataset.save_to_disk(subset_evaluation_result_path, storage_options={})
    score_dataset.push_to_hub(
        "CharlyR/varbench-evaluation",
        config_name=subset,
        split=full_config_name,
    )
