from datasets import Dataset
from loguru import logger

from varbench.evaluation.clip_comparer import ClipComparer
from varbench.evaluation.line_patch_scorer import compute_line_score
from varbench.utils.patches import patches


class Metric():
    def __init__(
        self, *args, **kwargs
    ) -> None:
        """instantiates a metric

        Args:
            k (int): The passK metric
        """
        pass

    def compute(self, dataset: Dataset) -> list:
        """computes the metric using the dataset
        The dataset should have columns: id,code,predictions,patches,result_description,image_solution,image_input,images_result

        Args:
            dataset (Dataset): The dataset used to copute the metric on
        """
        pass


class PatchMetric(Metric):
    def compute(self, dataset: Dataset):
        logger.info("Computing patch_score")
        inputs = dataset["code"]
        predictions = dataset["predictions"]
        patch_list = dataset["patches"]
        individual_patches = [patches(i, p) for i, p in zip(inputs, predictions)]
        individual_patches_scores = [
            bool(set(i) & set(d)) for i, d in zip(individual_patches, patch_list)
        ]
        return individual_patches_scores


class LineMetric(Metric):
    def compute(self, dataset: Dataset):
        logger.info("Computing line_score")
        inputs = dataset["code"]
        predictions = dataset["predictions"]
        patch_list = dataset["patches"]
        individual_patches = [patches(i, p) for i, p in zip(inputs, predictions)]
        individual_lines_scores = compute_line_score(individual_patches, patch_list)

        return individual_lines_scores


class ClipImageMetric(Metric):
    def __init__(self, clip_comparer: ClipComparer = None, *args, **kwargs) -> None:
        self.clip_comparer = clip_comparer
        super().__init__(*args, **kwargs)
    def compute(self, dataset: Dataset):
        logger.info("Computing clip image to image similarity scores")
        image_result = dataset["images_result"]
        image_solution = dataset["image_solution"]
        individual_image_scores = self.clip_comparer.image_similarities(
            image_result, image_solution
        )
        return individual_image_scores


class ClipTextMetric(Metric):
    def __init__(self, clip_comparer: ClipComparer = None, *args, **kwargs) -> None:
        self.clip_comparer = clip_comparer
        super().__init__(*args, **kwargs)
    def compute(self, dataset: Dataset):
        logger.info("Computing clip text to image similarity scores")
        image_result = dataset["images_result"]
        result_description = dataset["result_description"]
        individual_text_scores = self.clip_comparer.text_similarities(
            image_result, result_description
        )
        return individual_text_scores


def instantiate_metrics(metric_names:list[str])->list[Metric]:
    metric_map={
        "patch":PatchMetric,
        "line":LineMetric,
        "clipImage":ClipImageMetric,
        "clipText":ClipTextMetric
    }
    metrics:set[Metric] = set([metric_map[m_name] for m_name in set(metric_names)])
    if set([ClipImageMetric,ClipTextMetric]) & metrics:
        clip_comparer = ClipComparer()
    return [metric(clip_comparer) for metric in metrics]