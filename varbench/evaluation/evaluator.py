import datasets

from varbench.evaluation.metrics import Metric
from varbench.utils.patches import patches
from ..prompt_templates import *
from ..agents.agent import Agent
from ..renderers import Renderer, RendererException
from PIL import Image
from ..utils.parsing import get_first_code_block


def generate(
    subset: datasets.Dataset, agent: Agent, renderer: Renderer
) -> datasets.Dataset:
    # computing the results
    original_predictions = agent.batchCompute(
        subset["instruction"], subset["code"], subset["id"], subset["image_input"]
    )
    
    subset_processed: datasets.Dataset = subset.add_column(
        "original_predictions", original_predictions, feature=datasets.Sequence(datasets.Value("string"))
    )
    
    # getting the code from the result predictions
    predictions = [
        [
            get_first_code_block(prediction)
            for prediction in row_predictions
            if get_first_code_block(prediction)
        ]
        for row_predictions in original_predictions
    ]

    subset_processed: datasets.Dataset = subset_processed.add_column(
        "predictions", predictions, feature=datasets.Sequence(datasets.Value("string"))
    )

    # computing the images out of the results
    images_lists = _images(predictions, renderer)
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
    )  # add_column does not support image lists, so using concatenation
    return subset_processed


def evaluate(subset: datasets.Dataset, metrics: list[Metric]) -> datasets.Dataset:
    predictions = subset["predictions"]
    pass_size = len(predictions[0])  # getting the number of k for the pass@k

    images_lists = subset["images_result"]

    # default metrics computation(parsing and compiling)
    individual_parsing_scores = [
        len(prediction_n) / pass_size for prediction_n in predictions
    ]
    prediction_pass_len = len(predictions[0])
    individual_compiling_scores = [
        (len(image_list) / prediction_pass_len) if prediction_pass_len != 0 else 0
        for image_list in images_lists
    ]

    subset: datasets.Dataset = subset.add_column(
        "parsing_score", individual_parsing_scores
    )
    subset: datasets.Dataset = subset.add_column(
        "compiling_score", individual_compiling_scores
    )
    # computing the patches
    individual_patches = [patches(i, p) for i, p in zip(subset["code"], predictions)]
    subset: datasets.Dataset = subset.add_column(
        "predictions_patches",
        individual_patches,
        feature=datasets.Sequence(datasets.Value("string")),
    )
    # computing metrics
    metric_results = [metric.compute(subset) for metric in metrics]

    for metric, metric_result in zip(metrics, metric_results):
        metric_results = [
            [round(i_metric_result, 5) for i_metric_result in result]
            for result in metric_result
        ]

        subset: datasets.Dataset = subset.add_column(
            type(metric).__name__,
            metric_result,
            feature=datasets.Sequence(datasets.Value("float")),
        )

    return subset


def _images(
    predictions: list[list[str]], renderer: Renderer
) -> list[list[Image.Image]]:

    new_width = 300  # TODO make it a config parameter
    new_height = 300

    output_images: list[list[Image.Image]] = []
    for row_predictions in predictions:
        row_output_images = []
        for prediction in row_predictions:
            try:
                result_image = renderer.from_string_to_image(prediction)
                result_image = result_image.resize((new_width, new_height))
                row_output_images.append(result_image)
            except RendererException:
                pass
        output_images.append(row_output_images)

    return output_images
