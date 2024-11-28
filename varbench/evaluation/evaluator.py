import difflib
import datasets
import subprocess
import os

from varbench.evaluation.clip_comparer import ClipComparer
from varbench.evaluation.metrics import Metric, MetricPolicy
from ..prompt_templates import *
from ..agent import Agent
from loguru import logger
from ..renderers import Renderer, RendererException
from PIL.Image import Image
import torch
import pandas as pd
import re
from .line_patch_scorer import compute_line_score
from ..utils.parsing import get_first_code_block
from ..utils.patches import patches


def evaluate(
    subset: datasets.Dataset, agent: Agent, renderer: Renderer, metrics: list[Metric]
):

    # computing the results
    predictions = agent.batchCompute(
        subset["instruction"], subset["code"], subset["id"], subset["image_input"]
    )
    pass_size = len(predictions[0])  # getting the number of k for the pass@k

    # getting the code from the result predictions
    predictions = [
        [
            get_first_code_block(prediction)
            for prediction in row_predictions
            if get_first_code_block(prediction)
        ]
        for row_predictions in predictions
    ]

    # computing the images out of the results
    images_lists = _images(predictions, renderer)

    subset_processed: datasets.Dataset = subset.add_column("predictions", predictions)
    subset_processed: datasets.Dataset = datasets.concatenate_datasets(
        [
            subset_processed,
            datasets.Dataset.from_dict(
                {"images_result": images_lists},
                features=datasets.Features(
                    {"images_result": datasets.Sequence(datasets.Image())}
                ),
            ),
        ],
        axis=1,
    )  # add_column does does support image lists, so using concatenation

    # default metrics computation(parsing and compiling)
    individual_parsing_scores = [
        len(prediction_n) / pass_size for prediction_n in predictions
    ]
    prediction_pass_len = len(predictions[0])
    individual_compiling_scores = [
        (len(image_list) / prediction_pass_len) if prediction_pass_len != 0 else 0
        for image_list in images_lists
    ]

    subset_processed: datasets.Dataset = subset_processed.add_column(
        "parsing_score", individual_parsing_scores
    )
    subset_processed: datasets.Dataset = subset_processed.add_column(
        "compiling_score", individual_compiling_scores
    )

    # computing metrics
    metric_results = [metric.compute(subset_processed) for metric in metrics]

    for metric, metric_result in zip(metrics, metric_results):
        subset_processed: datasets.Dataset = subset_processed.add_column(
            type(metric).__name__, metric_result
        )

    computed_metrics_names = [type(metric).__name__ for metric in metrics]

    # each metric is computed on list of predictions of length pass@k, and yields a list of list of result of the same length.
    # from that list[list[float]](the results), we get the best result according to a certain policy(here the arithmetic mean)
    subset_processed = subset_processed.map(
        compute_best_prediction, fn_kwargs={"computed_metrics_names": computed_metrics_names}
    )

    scores = {
        metric_name: sum(subset_processed[f"best_{metric_name}"])
        / len(subset_processed)
        for metric_name in computed_metrics_names
    }
    return scores, subset_processed


def _images(predictions: list[list[str]], renderer: Renderer) -> list[list[Image]]:
    output_images: list[list[Image]] = []
    for row_predictions in predictions:
        row_output_images = []
        for prediction in row_predictions:
            try:
                result_image = renderer.from_string_to_image(prediction)
                row_output_images.append(result_image)
            except RendererException:
                pass
        output_images.append(row_output_images)
    return output_images


def compute_best_prediction(row, computed_metrics_names: list[str]):
    """Computes the best prediction out of arrays of metrics, according to a policy(arithmetic, geometrical, or harmonic mean)

    Args:
        row (_type_): the row to make the treatment on
        metrics (list[Metric]): the list of metrics to compute the best prediction on

    """
    scores_predictions_array = []
    computed_metric_amount = len(
        row[computed_metrics_names[0]]
    )  # assuming all metrics computed the same amount of scores
    for i in range(computed_metric_amount):
        current_score_array = []
        for metric_name in computed_metrics_names:
            current_score_array.append(row[metric_name][i])
        scores_predictions_array.append(current_score_array)

    policy_applied_array = [
        MetricPolicy.mathematical_average(current_scores)
        for current_scores in scores_predictions_array
    ]
    
    if len(policy_applied_array)== 0:
        #nothing was able to be computed from the predictions
        row["var_score"] = 0
        row["index_best_prediction"] = -1
        for metric_name in computed_metrics_names:
            row[f"best_{metric_name}"] = 0
        return row
    
    max_value = max(policy_applied_array)
    index_max_value = policy_applied_array.index(max_value)
    row["var_score"] = max_value
    row["index_best_prediction"] = index_max_value
    for metric_name in computed_metrics_names:
        row[f"best_{metric_name}"] = row[metric_name][index_max_value]
    return row
