import unittest

from varbench.evaluation.metrics import (
    FeatureMatchMetric,
    LPIPSMetric,
    MSSSIMMetric,
    Metric,
    PSNRMetric,
)
from PIL import Image

import datasets
from loguru import logger


class TestMetrics(unittest.TestCase):

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
