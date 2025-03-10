import difflib
from varbench.dataset_workflow.utils import unify_code
from loguru import logger

def single_patch(input: str, prediction: str) -> str:
    """Generates a patch from the given input(which will be splitted) and the prediction"""
    solution_split = prediction.splitlines()
    input_split = input.splitlines()
    current_diff = "\n".join(
        list(difflib.unified_diff(input_split, solution_split, n=0))[2:]
    )
    return current_diff


def patches(input, predictions: list[str]) -> list:
    return [single_patch(input, prediction) for prediction in predictions]
