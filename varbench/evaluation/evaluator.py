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
from .line_diff_scorer import compute_line_score
from ..utils.parsing import get_first_code_block
from ..utils.diffs import diffs



def evaluate(subset: Dataset, model: LLM_Model, compiler: Compiler):

    subset_processed = subset.map(create_message_row)

    predictions = model.batchRequest(
        subset_processed["messages"], subset_processed["id"]
    )

    subset_processed: Dataset = subset_processed.add_column("predictions", predictions)
    logger.info(subset_processed["messages"])
    logger.info(subset_processed["predictions"])
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
            "content": SYSTEM_PROMPT_GENERATION,
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
    diffs: list[list[str]],
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

    pass_size = len(predictions[0])  # getting the number of k for the pass@k
    dataset_lenght = len(image_solutions)
    # getting the code from the predictions
    predictions = [
        [
            get_first_code_block(prediction)
            for prediction in row_predictions
            if get_first_code_block(prediction)
        ]
        for row_predictions in predictions
    ]

    # computing parsing score
    individual_parsing_scores = [
        len(prediction_n) / pass_size for prediction_n in predictions
    ]

    # first computing if any of the diffs are in the prediction, if so the score is 1 for the row
    individual_diffs = [diffs(i, p) for i, p in zip(inputs, predictions)]
    logger.info(individual_diffs)
    individual_diffs_scores = [
        bool(set(i) & set(d)) for i, d in zip(individual_diffs, diffs)
    ]

    # getting line scores
    individual_lines_scores = compute_line_score(individual_diffs, diffs)

    # computing the clip similarity between compiled images and descriptions of results, as well as the image similarities
    images_lists = _images(predictions)
    prediction_pass_len = len(predictions[0])
    individual_compiling_scores = [
        (len(image_list) / prediction_pass_len) if prediction_pass_len != 0 else 0
        for image_list in images_lists
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

    diff_score = sum(individual_diffs_scores) / dataset_lenght
    text_score = sum(individual_text_scores) / dataset_lenght
    image_score = sum(individual_image_scores) / dataset_lenght
    compiling_score = sum(individual_compiling_scores) / dataset_lenght
    parsing_score = sum(individual_parsing_scores) / dataset_lenght
    line_score = sum(individual_lines_scores) / dataset_lenght

    varscores = [
        d if d else (t + i) / 2
        for d, t, i in zip(
            individual_diffs_scores,
            individual_text_scores,
            individual_image_scores,
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
            "individual_parsing_scores": individual_parsing_scores,
            "individual_lines_scores": individual_lines_scores,
            "result_description": result_descriptions,
            "individual_diffs":individual_diffs,
            "id": ids,
            "predictions": predictions,
        }
    )

    return {
        "line_score": line_score,
        "diffs_score": diff_score,
        "varscore": varscore,
        "text_scores": text_score,
        "image_score": image_score,
        "compiling_score": compiling_score,
        "parsing_score": parsing_score,
    }, output_dataset
