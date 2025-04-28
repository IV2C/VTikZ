from datasets import Dataset
from loguru import logger
import torch.utils
import torch.utils

from vtikz.evaluation.clip_comparer import ClipComparer
from vtikz.evaluation.eed.EED import eed
from vtikz.evaluation.line_patch_scorer import compute_line_score
from vtikz.utils.patches import patches


class Metric:
    def __init__(self, *args, **kwargs) -> None:
        """instantiates a metric"""
        pass

    def compute(self, dataset: Dataset) -> list[list[list[float]]]:
        """computes the metric using the dataset
        The dataset should have columns: id,code,code_solution,predictions,predictions_patches,patches,result_description,image_solution,image_input,images_result

        Args:
            dataset (Dataset): The dataset used to compute the metric on

        Returns:
            a list rows, each row containing a list of comparisons of generated with references codes.
        """
        pass


# NEW
from .template import template_valid


class TemplateMetric(Metric):
    def compute(self, dataset: Dataset):
        logger.info("Computing template_score")
        template_solution_code = dataset["template_solution_code"]
        predictions = dataset["predictions"]
        individual_template_scores = [
            [[100*int(template_valid(tem, pred)) for pred in preds] for tem in templates]
            for templates, preds in zip(template_solution_code, predictions)
        ]
        return individual_template_scores


class ImageEqualityMetric(Metric):
    def __init__(self, *args, **kwargs):
        import torchvision.transforms as transforms

        self.to_tensor = transforms.ToTensor()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:

        def equality(hyp, ref):
            return 100 if torch.equal(hyp, ref) else 0

        references = dataset["image_solution"]
        predictions = dataset["images_result"]
        references = [
            [self.to_tensor(pil_image.convert("RGB")) for pil_image in pil_images]
            for pil_images in references
        ]
        predictions = [
            [self.to_tensor(pil_image.convert("RGB")) for pil_image in pil_images]
            for pil_images in predictions
        ]

        return [
            [
                [equality(reference, prediction) for prediction in row_predictions]
                for reference in refs
            ]
            for refs, row_predictions in zip(references, predictions)
        ]


class LineMetric(Metric):
    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing line_score")
        patch = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        individual_lines_scores = compute_line_score(individual_patches, patch)

        return individual_lines_scores


class CrystalBleuPatchMetric(Metric):
    def __init__(self, dataset: Dataset, *args, **kwargs) -> None:
        from .crystal_bleu import CrystalBLEU

        full_corpus: list[str] = dataset["code"]

        self.crystal_bleu = CrystalBLEU(
            effective_order=True, max_ngram_order=8, full_corpus=full_corpus, k=500
        )
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing crystalbleu_patch_score")
        patches = dataset["patch"]
        prediction_patches = dataset["predictions_patches"]
        bleu_patch_scores = [
            [
                [
                    self.crystal_bleu.sentence_score(row_patch, [ref_patch]).score
                    for row_patch in computed_patches
                ]
                for ref_patch in reference_patches
            ]
            for computed_patches, reference_patches in zip(prediction_patches, patches)
        ]
        return bleu_patch_scores


# FORMER


class PatchMetric(Metric):
    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing patch_score")
        patch = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        individual_patches_scores = [
            [int(computed_patch == p) * 100.0 for computed_patch in i]
            for i, p in zip(individual_patches, patch)
        ]
        return individual_patches_scores


class BleuMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import BLEU

        self.bleu = BLEU(effective_order=True, max_ngram_order=8)
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing bleu_score")
        all_predictions = dataset["predictions"]
        solutions = dataset["code_solution"]

        bleu_scores = [
            [
                self.bleu.sentence_score(row_prediction, [solution]).score
                for row_prediction in predictions
            ]
            for predictions, solution in zip(all_predictions, solutions)
        ]

        return bleu_scores


class BleuPatchMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import BLEU

        self.bleu = BLEU(effective_order=True, max_ngram_order=8)
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing bleu_patch_score")
        patches = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        bleu_patch_scores = [
            [
                self.bleu.sentence_score(row_patch, [reference_patch]).score
                for row_patch in computed_patches
            ]
            for computed_patches, reference_patch in zip(individual_patches, patches)
        ]
        return bleu_patch_scores


class ChrfMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import CHRF

        self.chrf = CHRF()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing chrf_score")
        all_predictions = dataset["predictions"]
        solutions = dataset["code_solution"]

        chrf_scores = [
            [
                self.chrf.sentence_score(row_prediction, [solution]).score
                for row_prediction in predictions
            ]
            for predictions, solution in zip(all_predictions, solutions)
        ]

        return chrf_scores


class ChrfPatchMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import CHRF

        self.chrf = CHRF()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing chrf_patch_score")
        patches = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        bleu_patch_scores = [
            [
                self.chrf.sentence_score(row_patch, [reference_patch]).score
                for row_patch in computed_patches
            ]
            for computed_patches, reference_patch in zip(individual_patches, patches)
        ]
        return bleu_patch_scores


class TERMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import TER

        self.ter = TER()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing TER_score")
        all_predictions = dataset["predictions"]
        solutions = dataset["code_solution"]

        ter_inverted_scores = [
            [
                (100 * 100)
                / (
                    100
                    + min(
                        100.0,
                        self.ter.sentence_score(row_prediction, [solution]).score,
                    )
                )
                for row_prediction in predictions
            ]
            for predictions, solution in zip(all_predictions, solutions)
        ]

        return ter_inverted_scores


class TERPatchMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from sacrebleu import TER

        self.ter = TER()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing ter_patch_score")
        patches = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        bleu_patch_scores = [
            [
                (100 * 100)
                / (
                    100
                    + min(
                        100.0,
                        self.ter.sentence_score(row_patch, [reference_patch]).score,
                    )
                )
                for row_patch in computed_patches
            ]
            for computed_patches, reference_patch in zip(individual_patches, patches)
        ]
        return bleu_patch_scores


class EEDMetric(Metric):

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing eed_score")
        all_predictions = dataset["predictions"]
        solutions = dataset["code_solution"]

        eed_inverted_scores = [
            [
                100 / (1 + eed(row_prediction, solution))
                for row_prediction in predictions
            ]
            for predictions, solution in zip(all_predictions, solutions)
        ]
        return eed_inverted_scores


class EEDPatchMetric(Metric):

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing eed_patch_score")
        patches = dataset["patch"]
        individual_patches = dataset["predictions_patches"]
        eed_patch_scores = [
            [
                100 / (1 + eed(row_patch, reference_patch))
                for row_patch in computed_patches
            ]
            for computed_patches, reference_patch in zip(individual_patches, patches)
        ]
        return eed_patch_scores


## Non-agnostic metric
class CrystalBleuMetric(Metric):
    def __init__(self, dataset: Dataset, *args, **kwargs) -> None:
        from .crystal_bleu import CrystalBLEU

        full_corpus: list[str] = dataset["code"]

        self.crystal_bleu = CrystalBLEU(
            effective_order=True, max_ngram_order=8, full_corpus=full_corpus, k=500
        )
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing crystalbleu_score")
        all_predictions = dataset["predictions"]
        solutions = dataset["code_solution"]

        bleu_scores = [
            [
                self.crystal_bleu.sentence_score(row_prediction, [solution]).score
                for row_prediction in predictions
            ]
            for predictions, solution in zip(all_predictions, solutions)
        ]

        return bleu_scores


########################################Image based metrics########################################


class ClipImageMetric(Metric):
    def __init__(self, clip_comparer: ClipComparer = None, *args, **kwargs) -> None:
        self.clip_comparer = clip_comparer
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
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

    def compute(self, dataset: Dataset) -> list[list[float]]:
        logger.info("Computing clip text to image similarity scores")
        image_result = dataset["images_result"]
        result_description = dataset["result_description"]
        individual_text_scores = self.clip_comparer.text_similarities(
            image_result, result_description
        )
        return individual_text_scores


import cv2
import numpy as np


class FeatureMatchMetric(Metric):

    def __init__(self, *args, **kwargs) -> None:

        self.sift = cv2.SIFT.create()
        self.bf_matcher = cv2.BFMatcher()
        self.threshold = 0.5
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        def _convert(image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

        def _filter_matches(matches):
            filtered_matches = []
            for m, n in matches:
                if m.distance < self.threshold * n.distance:
                    filtered_matches.append(m)
            return filtered_matches

        def compare_sift_brut(
            reference: cv2.typing.MatLike, predictions: list[cv2.typing.MatLike]
        ) -> list[float]:
            ref_features = self.sift.detectAndCompute(reference, None)[1]
            predictions_features = [
                self.sift.detectAndCompute(prediction, None)[1]
                for prediction in predictions
            ]
            ref_own_matches = _filter_matches(
                self.bf_matcher.knnMatch(ref_features, ref_features, k=2)
            )  # theoric maximum amount of matches that a image can have with the reference
            pred_ref_matches = [
                _filter_matches(
                    self.bf_matcher.knnMatch(pred_features, ref_features, k=2)
                )
                for pred_features in predictions_features
            ]

            nb_max_matches = len(ref_own_matches)

            pred_scores = [
                100.0 * (len(pred_matches) / nb_max_matches)
                for pred_matches in pred_ref_matches
            ]

            return pred_scores

        # converting to cv2 images
        reference_images = [
            _convert(ref_image) for ref_image in dataset["image_solution"]
        ]
        predictions_images = [
            [_convert(result_image) for result_image in row_images]
            for row_images in dataset["images_result"]
        ]

        return [
            compare_sift_brut(ref_image, predictions)
            for ref_image, predictions in zip(reference_images, predictions_images)
        ]


class LPIPSMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        import torchvision.transforms as transforms
        import lpips

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.loss_fn_alex = lpips.LPIPS(net="alex")
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:
        references = dataset["image_solution"]
        predictions = dataset["images_result"]
        scores = []

        for reference_image, predicted_images in zip(references, predictions):
            # Transform the reference and predictions
            reference_tensor = self.transform(reference_image.convert("RGB")).unsqueeze(
                0
            )
            # Compute loss for each predicted image individually
            individual_scores = [
                100.0
                - 100.0
                * self.loss_fn_alex(
                    reference_tensor, self.transform(img.convert("RGB")).unsqueeze(0)
                ).item()
                for img in predicted_images
            ]
            scores.append(individual_scores)

        return scores


class PSNRMetric(Metric):
    def compute(self, dataset: Dataset) -> list[list[float]]:

        def PSNR(original, compressed):
            mse = np.mean((original - compressed) ** 2)
            if mse == 0:
                return 100
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            return psnr

        references = dataset["image_solution"]
        predictions = dataset["images_result"]
        references = [np.array(pil_image.convert("RGB")) for pil_image in references]
        predictions = [
            [np.array(pil_image.convert("RGB")) for pil_image in pil_images]
            for pil_images in predictions
        ]

        return [
            [PSNR(reference, prediction) for prediction in row_predictions]
            for reference, row_predictions in zip(references, predictions)
        ]


class MSSSIMMetric(Metric):
    def __init__(self, *args, **kwargs) -> None:
        from pytorch_msssim import MS_SSIM
        from torchvision.transforms import ToTensor

        self.totensor = ToTensor()
        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=False, channel=3)

        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:

        references = dataset["image_solution"]
        predictions = dataset["images_result"]
        references = [
            self.totensor(pil_image.convert("RGB")).unsqueeze(0)
            for pil_image in references
        ]
        predictions = [
            [
                self.totensor(pil_image.convert("RGB")).unsqueeze(0)
                for pil_image in pil_images
            ]
            for pil_images in predictions
        ]

        return [
            [
                100.0 * self.ms_ssim_module(reference, prediction).item()
                for prediction in row_predictions
            ]
            for reference, row_predictions in zip(references, predictions)
        ]


class MSEMetric(Metric):
    def __init__(self, *args, **kwargs):
        import torchvision.transforms as transforms

        self.to_tensor = transforms.ToTensor()
        super().__init__(*args, **kwargs)

    def compute(self, dataset: Dataset) -> list[list[float]]:

        def mse_similarity(hyp, ref):
            err = torch.mean((ref - hyp) ** 2)

            return 100.0 - (100.0 * err)

        references = dataset["image_solution"]
        predictions = dataset["images_result"]
        references = [
            self.to_tensor(pil_image.convert("RGB")) for pil_image in references
        ]
        predictions = [
            [self.to_tensor(pil_image.convert("RGB")) for pil_image in pil_images]
            for pil_images in predictions
        ]

        return [
            [
                mse_similarity(reference, prediction).item()
                for prediction in row_predictions
            ]
            for reference, row_predictions in zip(references, predictions)
        ]


agnostic_metric_map = {
    "patch": PatchMetric,
    "line": LineMetric,
    "clipImage": ClipImageMetric,
    "clipText": ClipTextMetric,
    "bleu": BleuMetric,
    "bleuPatch": BleuPatchMetric,
    "chrf": ChrfMetric,
    "chrfPatch": ChrfPatchMetric,
    "TER": TERMetric,
    "TERPatch": TERPatchMetric,
    "featureMatch": FeatureMatchMetric,
    "LPIPS": LPIPSMetric,
    "psnr": PSNRMetric,
    "msssim": MSSSIMMetric,
    "MSE": MSEMetric,
    "Template": TemplateMetric,
    "ImageEquality": ImageEqualityMetric,
}
non_agnostic_metric_map = {
    "crystalBleu": CrystalBleuMetric,
    "crystalBleuPatch": CrystalBleuPatchMetric,
}


def instantiate_agnostic_metrics(metric_names: list[str]) -> list[Metric]:
    """instantiates the appropriate metrics, given alist of strings

    Args:
        metric_names (list[str]): list of the metrics' names
        dataset (Dataset): The hg dataset from the current evaluated subset

    Returns:
        list[Metric]: list of metrics
    """
    metrics: set[Metric] = set(
        [
            agnostic_metric_map[m_name]
            for m_name in set(metric_names)
            if m_name in agnostic_metric_map
        ]
    )
    logger.info(f"loading metrics : " + str([metric.__name__ for metric in metrics]))
    hasClip = set([ClipImageMetric, ClipTextMetric]) & metrics
    if hasClip:
        clip_comparer = ClipComparer()
        return [metric(clip_comparer) for metric in metrics]
    else:
        return [metric() for metric in metrics]

def instantiate_non_agnostic_metrics(
    metric_names: list[str], dataset: Dataset
) -> list[Metric]:
    """isntantiates the appropriate metrics, given alist of strings

    Args:
        metric_names (list[str]): list of the metrics' names
        dataset (Dataset): The hg dataset from the current evaluated subset

    Returns:
        list[Metric]: list of metrics
    """

    metrics: set[Metric] = set(
        [
            non_agnostic_metric_map[m_name]
            for m_name in set(metric_names)
            if m_name in non_agnostic_metric_map
        ]
    )
    logger.warning(dataset)
    logger.info(
        f"loading non agnostic metrics : "
        + str([metric.__name__ for metric in metrics])
    )
    return [metric(dataset=dataset) for metric in metrics]


import math
