import difflib


def single_patch(input: str, prediction: list[str]) -> str:
    """Generates a patch from the given input(which will be splitted) and the prediction"""
    return "".join(list(difflib.unified_diff(input.split("\n"), prediction, n=0))[2:])


def patches(input, predictions: list[str]) -> list:
    predictions = [prediction.split("\n") for prediction in predictions]
    return [single_patch(input, prediction) for prediction in predictions]
