import unittest

from varbench.evaluation.metrics import (
    BleuPatchMetric,
    ChrfPatchMetric,
    FeatureMatchMetric,
    LPIPSMetric,
    MSSSIMMetric,
    Metric,
    PSNRMetric,
    PatchMetric,
    TERPatchMetric,
)
from PIL import Image

import datasets
from loguru import logger


class TestImageMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        dataset = {
            "image_solution": [Image.open("tests/resources/images/dog.jpeg")],
            "images_result": [
                [
                    Image.open("tests/resources/images/dog.jpeg"),
                    Image.open("tests/resources/images/dog-redeye.jpeg"),
                    Image.open("tests/resources/images/dog-rotated.jpeg"),
                ]
            ],
        }
        features = datasets.Features(
            {
                "image_solution": datasets.Image(),
                "images_result": datasets.Sequence(datasets.Image()),
            }
        )
        self.dataset = datasets.Dataset.from_dict(dataset, features=features)

    def test_feature_match(self):
        feature_match_metric: Metric = FeatureMatchMetric()
        result_scores = feature_match_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_lpips(self):
        Lpips_metric: Metric = LPIPSMetric()

        result_scores = Lpips_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_psnr(self):
        psnr_metric: Metric = PSNRMetric()

        result_scores = psnr_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_msssim(self):
        msssim_metric: Metric = MSSSIMMetric()

        result_scores = msssim_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])


class TestPatchTextMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        dataset = {
            "patch": ["@@ -1 +1 @@ \n -Hello Word! \n +Hello World!"],
            "predictions_patches": [
                [
                    "@@ -1 +1 @@ \n -Hello Word! \n +Hello World!",
                    "@@ -1 +1 @@ \n -Hello Word! \n +Hello Worlds!",
                    "test",
                ]
            ],
        }
        features = datasets.Features(
            {
                "patch": datasets.Value("string"),
                "predictions_patches": datasets.Sequence(datasets.Value("string")),
            }
        )
        self.dataset = datasets.Dataset.from_dict(dataset, features=features)

    def test_patch_metrics(self):
        patch_metric: Metric = PatchMetric()
        result_scores = patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertEqual(result_scores[0][1], 0.0)
        self.assertEqual(result_scores[0][2], 0.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_bleu_patch_metrics(self):
        bleu_patch_metric: Metric = BleuPatchMetric()
        result_scores = bleu_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 85.55262)
        self.assertEqual(round(result_scores[0][2], 5), 0.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_chrf_patch_metrics(self):
        chrf_patch_metric: Metric = ChrfPatchMetric()
        result_scores = chrf_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 96.33817)
        self.assertEqual(round(result_scores[0][2], 5), 0.97656)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_TER_patch_metrics(self):
        ter_patch_metric: Metric = TERPatchMetric()
        result_scores = ter_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 88.88889)
        self.assertEqual(round(result_scores[0][2], 5), 50.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])
