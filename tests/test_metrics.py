import unittest

from vtikz.evaluation.metrics import (
    BleuMetric,
    BleuPatchMetric,
    ChrfMetric,
    ChrfPatchMetric,
    CrystalBleuMetric,
    CrystalBleuPatchMetric,
    EEDMetric,
    EEDPatchMetric,
    FeatureMatchMetric,
    ImageEqualityMetric,
    LPIPSMetric,
    LineMetric,
    MSSSIMMetric,
    Metric,
    PSNRMetric,
    PatchMetric,
    TERMetric,
    TERPatchMetric,
    MSEMetric,
    TemplateMetric,
)
from PIL import Image

import datasets
from loguru import logger


class TestNewDatasetMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        # dataset for test purpose, columns relations not respected
        dataset = {
            "image_solution": [
                [
                    Image.open("tests/resources/images/dog.jpeg"),
                    Image.open("tests/resources/images/dog-redeye.jpeg"),
                ]
            ],
            "images_result": [
                [
                    Image.open("tests/resources/images/dog.jpeg"),
                    Image.open("tests/resources/images/dog-redeye.jpeg"),
                    Image.open("tests/resources/images/dog-rotated.jpeg"),
                ]
            ],
            "code": ["@@ @@ @@ @@ Hello Hello Hello Hello"],
            "patch": [
                [
                    "@@ -1 +1,2 @@ \n -Hello Word! \n +Hello World!",
                    "@@ -1 +1 @@ \n -Hello Word! \n +Hello Worlds!",
                ]
            ],
            "predictions_patches": [
                [
                    "@@ -1,0 +1,2 @@ \n -Hello Word! \n +Hello World!",
                    "@@ -1 +1 @@ \n -Hello Word! \n +Hello Worlds!",
                    "@@ -3 +3 @@ test",
                ]
            ],
            "predictions": [
                [
                    """
\\definecolor{blue}{rgb}{1}
\\definecolor{blue}{rgb}{1}
""",
                    """
\\definecolor{blue}{rgb}{5}
\\definecolor{blue}{rgb}{6}
""",
                ]
            ],
            "template_solution_code": [
                [
                    """
\\definecolor{blue}{rgb}{§choice([1,2],1)}
\\definecolor{blue}{rgb}{§range(1,2,1)}
""",
                    """
\\definecolor{blue}{rgb}{§choice([5,6],5)}
\\definecolor{blue}{rgb}{§range(5,6,5)}
""",
                ]
            ],
        }
        features = datasets.Features(
            {
                "image_solution": datasets.Sequence(datasets.Image()),
                "images_result": datasets.Sequence(datasets.Image()),
                "code": datasets.Value("string"),
                "patch": datasets.Sequence(datasets.Value("string")),
                "predictions_patches": datasets.Sequence(datasets.Value("string")),
                "template_solution_code": datasets.Sequence(datasets.Value("string")),
                "predictions": datasets.Sequence(datasets.Value("string")),
            }
        )
        self.dataset = datasets.Dataset.from_dict(dataset, features=features)

    def test_Image_equality(self):
        image_equality_metric: Metric = ImageEqualityMetric()
        result_scores = image_equality_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0], [[100, 0, 0], [0, 100, 0]])

    def test_crystalbleu_patch_metrics(self):
        ter_patch_metric: Metric = CrystalBleuPatchMetric(dataset=self.dataset)
        result_scores = ter_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        for i, row in enumerate(
            [
                [71.20952307293861, 30.661487102926746, 1.2614755640151063],
                [26.88927578056405, 100.00000000000004, 1.2614755640151063],
            ]
        ):
            for j, expected in enumerate(row):
                self.assertAlmostEqual(result_scores[0][i][j], expected, places=5)

    def test_line_metrics(self):
        ter_patch_metric: Metric = LineMetric(dataset=self.dataset)
        result_scores = ter_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        for i, row in enumerate([[50.0, 50.0, 0.0], [0, 100.0, 0.0]]):
            for j, expected in enumerate(row):
                self.assertAlmostEqual(result_scores[0][i][j], expected, places=5)

    def test_template_metrics(self):
        ter_patch_metric: Metric = TemplateMetric(dataset=self.dataset)
        result_scores = ter_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        for i, row in enumerate([[100, 0], [0, 100]]):
            for j, expected in enumerate(row):
                self.assertAlmostEqual(result_scores[0][i][j], expected, places=5)


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

    def test_mse(self):
        mse_metric: Metric = MSEMetric()

        result_scores = mse_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(result_scores[0][0], 100.0)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])


class TestPatchTextMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        dataset = {
            "code": ["@@ @@ @@ @@ Hello Hello Hello Hello"],
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
                "code": datasets.Value("string"),
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
        self.assertEqual(round(result_scores[0][1], 5), 80.62626)
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

    def test_EED_patch_metrics(self):
        eed_patch_metric: Metric = EEDPatchMetric()
        result_scores = eed_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 98.88393)
        self.assertEqual(round(result_scores[0][2], 5), 51.81159)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])


class TestTextMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        dataset = {
            "code": [open("tests/resources/tikz/cow/cow.tex").read()],
            "code_solution": [
                open("tests/resources/tikz/cow/cow_brown_dots.tex").read()
            ],
            "predictions": [
                [
                    open("tests/resources/tikz/cow/cow_brown_dots.tex").read(),
                    open("tests/resources/tikz/cow/cow_brown_dots_dif.tex").read(),
                ]
            ],
        }
        features = datasets.Features(
            {
                "code": datasets.Value("string"),
                "code_solution": datasets.Value("string"),
                "predictions": datasets.Sequence(datasets.Value("string")),
            }
        )
        self.dataset = datasets.Dataset.from_dict(dataset, features=features)

    def test_bleu_metrics(self):
        bleu_patch_metric: Metric = BleuMetric()
        result_scores = bleu_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 97.80731)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_chrf_metrics(self):
        chrf_patch_metric: Metric = ChrfMetric()
        result_scores = chrf_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 99.22962)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_TER_metrics(self):
        ter_patch_metric: Metric = TERMetric()
        result_scores = ter_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 94.73684)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_crystalbleu_metrics(self):
        crystalbleu_patch_metric: Metric = CrystalBleuMetric(dataset=self.dataset)
        result_scores = crystalbleu_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 85.43634)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])

    def test_eed_metrics(self):
        eed_patch_metric: Metric = EEDMetric(dataset=self.dataset)
        result_scores = eed_patch_metric.compute(self.dataset)
        logger.info(result_scores)
        self.assertEqual(round(result_scores[0][0], 5), 100.0)
        self.assertEqual(round(result_scores[0][1], 5), 98.8533)
        self.assertTrue(sorted(result_scores[0], reverse=True) == result_scores[0])
