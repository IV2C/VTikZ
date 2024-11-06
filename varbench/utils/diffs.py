import difflib


def single_diff(input:str, prediction:str) -> str:
    return "".join(list(difflib.unified_diff(input.split("\n"), prediction, n=0))[2:])


def diffs(input, predictions: list) -> list:
    predictions = [prediction.split("\n") for prediction in predictions]
    return [single_diff(input, prediction) for prediction in predictions]
