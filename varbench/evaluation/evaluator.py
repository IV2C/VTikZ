import datasets

from varbench.evaluation.metrics import Metric
from varbench.utils.patches import patches
from ..utils.prompts.simple_templates import *
from ..agents.agent import Agent
from ..renderers import Renderer, RendererException
from PIL import Image
from ..utils.parsing import get_first_code_block
from loguru import logger
from varbench.dataset_workflow.utils import unify_code


def generate(
    subset: datasets.Dataset, agent: Agent, renderer: Renderer
) -> datasets.Dataset:
    # computing the results
    original_predictions = agent.batchCompute(
        subset["instruction"], subset["code"], subset["id"], subset["image_input"]
    )

    subset_processed: datasets.Dataset = subset.add_column(
        "original_predictions",
        original_predictions,
        feature=datasets.Sequence(datasets.Value("string")),
    )

    # unify input codes
    def unify_solutions(row):
        row["template_solution_code"] = [
            unify_code(templated_code)
            for templated_code in row["template_solution_code"]
        ]
        row["code_solution"] = [
            unify_code(code_solution) for code_solution in row["code_solution"]
        ]
        row["code"] = unify_code(row["code"])
        return row

    subset_processed = subset_processed.map(unify_solutions)

    # getting the code from the result predictions
    predictions = [
        [
            unify_code(get_first_code_block(prediction))
            for prediction in row_predictions
            if get_first_code_block(prediction)
        ]
        for row_predictions in original_predictions
    ]

    subset_processed: datasets.Dataset = subset_processed.add_column(
        "predictions", predictions, feature=datasets.Sequence(datasets.Value("string"))
    )

    # computing the images out of the results
    indexes, images_lists = _images(predictions, renderer)
    subset_processed: datasets.Dataset = subset_processed.add_column(
        "image_result_indexes",
        indexes,
        feature=datasets.Sequence(datasets.Value("int32")),
    )
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
            [
                [round(metric_res, 5) for metric_res in i_metric_result]
                for i_metric_result in result
            ]
            for result in metric_result
        ]

        subset: datasets.Dataset = subset.add_column(
            type(metric).__name__,
            metric_result,
            feature=datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
        )

    subset: datasets.Dataset = _extend_metric_computations(subset)
    return subset


def _extend_metric_computations(dataset: datasets.Dataset) -> datasets.Dataset:
    """The image-based metrics in the dataset are only computed for x out of y code generated, because some of the code can't compile.
    During the compiling(method _images), we compute the indexes of the images that did compute and put it in an array.
    This method takes as input the dataset, find the names of the columns that contains image-based metrics, and extends the computed
    list with Nones in the places where the code could not render(be compiled into) an image
    """

    metrics_names = [name for name in dataset.column_names if "Metric" in name]
    potential_image_metrics_names = [
        name
        for name in metrics_names
        if any(
            len(row) < len(parsed)
            for row, parsed in zip(dataset[name], dataset["predictions_patches"])
        )
    ]  # named potential because if all images have been compiled without error we skip the process completely

    def _ext_none(row, col_name: str):
        "Extends the row with nones at unreferenced indexes"
        initial = [None] * len(row["predictions_patches"])
        for index, ar_value in zip(row["image_result_indexes"], row[col_name]):
            initial[index] = ar_value
        row[col_name] = initial
        return row

    def _ext_none_metric(row, col_name: str):
        "Extends the row with nones at unreferenced indexes"
        initial = [[None] * len(row["predictions"])] * len(row["code_solution"])
        for ind, sub_eval in enumerate(row[col_name]):
            for index, ar_value in zip(row["image_result_indexes"], sub_eval):
                initial[ind][index] = ar_value
        row[col_name] = initial
        return row

    dataset = dataset.map(_ext_none, fn_kwargs={"col_name": "images_result"})
    for metric_name in potential_image_metrics_names:
        dataset = dataset.map(_ext_none_metric, fn_kwargs={"col_name": metric_name})
    return dataset


def _images(
    predictions: list[list[str]], renderer: Renderer
) -> tuple[list[list[int]], list[list[Image.Image]]]:

    new_width = 300  # TODO make it a config parameter
    new_height = 300

    output_images_indexes: list[list[int]] = []
    output_images: list[list[Image.Image]] = []
    for row_predictions in predictions:
        row_output_images = []
        row_output_images_indexes = []
        for id, prediction in enumerate(row_predictions):
            try:
                result_image = renderer.from_string_to_image(prediction)
                result_image = result_image.resize((new_width, new_height))
                row_output_images.append(result_image)
                row_output_images_indexes.append(id)
            except RendererException:
                pass
        output_images.append(row_output_images)
        output_images_indexes.append(row_output_images_indexes)
    return output_images_indexes, output_images
