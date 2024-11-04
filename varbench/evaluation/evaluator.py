import difflib
from datasets import Dataset
import subprocess
import os

from varbench.evaluation.clip_comparer import ClipComparer
from ..prompt_templates import *
from ..model import LLM_Model
from loguru import logger
from ..compilers import Compiler, CompilerException
from PIL.Image import Image
import torch
import pandas as pd
import re


def _get_first_code_block(text):
    # Regular expression to find the first code block, ignoring the language specifier
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def evaluate(subset: Dataset, model: LLM_Model, compiler: Compiler):

    subset_processed = subset.map(create_message_row)

    predictions = model.batchRequest(
        subset_processed["messages"], subset_processed["id"]
    )
    predictions = [
        [_get_first_code_block(prediction) for prediction in row_predictions]
        for row_predictions in predictions
    ]
    subset_processed: Dataset = subset_processed.add_column("predictions", predictions)
    subset_processed.save_to_disk(".tmp/computed_dataset_" + model.model_name)

    return _compute(
        compiler,
        subset_processed["id"],
        subset_processed["code"],
        subset_processed["predictions"],
        subset_processed["diffs"],
        subset_processed["result_description"],
        subset_processed["image_solution"],
    )


def create_message_row(row):
    """Add a row that contains the prompt"""
    user_instruction = IT_PROMPT.format(
        instruction=row["instruction"], content=row["code"]
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {"role": "user", "content": user_instruction},
    ]

    row["messages"] = messages
    return row


def _compute(
    compiler: Compiler,
    ids: list[str],
    inputs: list[str],
    predictions: list[list[str]],
    diffs: list[str],
    result_descriptions: list[str],
    image_solutions: list[Image],
) -> tuple[object, pd.DataFrame]:
    """
    Computes the score.

    Args:
        compiler (Compiler): Compiler to convert text to images.
        ids (list[str]): List of dataset identifiers.
        inputs (list[str]): List of input code snippets.
        predictions (list[list[str]]): Model-generated predictions for fulfilling the instructions, organized as a list of lists for pass@k scoring.
        diffs (list[str]): List of code diffs that fulfill the instruction.
        descriptions (list[str]): List of descriptions of the expected results, used for scoring with a CLIP model.
        image_solutions (list[PIL.Image.Image]): List of solution images.
    """

    def _diffs(input, predictions: list) -> list:
        predictions = [prediction.split("\n") for prediction in predictions]
        return [
            "".join(list(difflib.unified_diff(input.split("\n"), prediction, n=0))[2:])
            for prediction in predictions
        ]

    def _images(predictions: list[list[str]]) -> list[list[Image]]:
        output_images: list[list[Image]] = []
        for row_predictions in predictions:
            row_output_images = []
            for prediction in row_predictions:
                try:
                    result_image = compiler.compile_from_string(prediction)
                    row_output_images.append(result_image)
                except CompilerException:
                    pass
            output_images.append(row_output_images)
        return output_images

    # first computing if any of the diffs are in the prediction, if so the score is 1 for the row
    individual_diffs_scores = [
        bool(set(_diffs(i, p)) & set(d)) for i, p, d in zip(inputs, predictions, diffs)
    ]

    # computing the clip similarity between compiled images and descriptions of results, as well as the image similarities
    images_lists = _images(predictions)
    prediction_pass_len = len(predictions[0])
    individual_compiling_scores = [
        len(image_list) / prediction_pass_len for image_list in images_lists
    ]

    clip_comparer: ClipComparer = ClipComparer(force_cpu=True)
    # TODO:Add a command line parameter for the clip parameters?

    individual_text_scores = clip_comparer.clip_scores(
        images_lists, result_descriptions
    )
    individual_image_scores = clip_comparer.image_similarities(
        images_lists, image_solutions
    )

    # individual_diffs_scores = {id: result for id, result in zip(ids, result_list)}

    diff_score = sum(individual_diffs_scores) / len(predictions)
    text_score = sum(individual_text_scores) / len(predictions)
    image_score = sum(individual_image_scores) / len(predictions)
    compiling_score = sum(individual_compiling_scores) / len(images_lists)

    varscores = [
        d if d else (t + i) / 2
        for d, t, i in zip(
            individual_diffs_scores, individual_text_scores, individual_image_scores
        )
    ]
    varscore = sum(varscores) / len(varscores)

    # logger.info(individual_diffs_scores)
    # logger.info(individual_text_scores)
    # logger.info(individual_image_scores)

    output_dataset: pd.DataFrame = pd.DataFrame.from_dict(
        {
            "individual_diffs_scores": individual_diffs_scores,
            "individual_text_scores": individual_text_scores,
            "individual_image_scores": individual_image_scores,
            "individual_compiling_scores": individual_compiling_scores,
            "id": ids,
        }
    )

    return {
        "diffs_score": diff_score,
        "varscore": varscore,
        "text_scores": text_score,
        "image_score": image_score,
        "compiling_score": compiling_score,
    }, output_dataset
