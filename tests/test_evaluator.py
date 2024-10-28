import unittest
import os
import timeout_decorator
from varbench.evaluation.evaluator import evaluate
from varbench.model import LLM_Model
from unittest.mock import MagicMock
from datasets import Dataset


class TestEvaluator(unittest.TestCase):

    def test_evaluator_metric_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}"
                ],
                "instruction": ["Rotate the line"],
                "diffs": [["@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0"]],
            }
        )

        # mock llm_model
        model: LLM_Model = LLM_Model()
        model.batchRequest = MagicMock(
            return_value=[
                "\\begin{tikzpicture} \\draw (0,1) -- (1,0); \\end{tikzpicture}"
            ]
        )

        # expected result
        expected = {"individual_scores": {"example1": True}, "varscore": 1.0}

        self.assertEqual(evaluate(dummy_dataset, model), expected)

    def test_evaluator_metric_not_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}"
                ],
                "instruction": ["Rotate the line"],
                "diffs": [["@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0"]],
            }
        )

        # mock llm_model
        model: LLM_Model = LLM_Model()
        model.batchRequest = MagicMock(return_value=["wrong_return_value"])

        # expected result
        expected = {"individual_scores": {"example1": False}, "varscore": 0.0}

        self.assertEqual(evaluate(dummy_dataset, model), expected)

    def test_evaluator_metric_exists_multiple(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1", "example2"],
                "code": [
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}",
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}",
                ],
                "instruction": ["Rotate the line", "Rotate the line"],
                "diffs": [
                    ["@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0"],
                    ["@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0"],
                ],
            }
        )

        # mock llm_model
        model: LLM_Model = LLM_Model()
        model.batchRequest = MagicMock(
            return_value=[
                "\\begin{tikzpicture} \\draw (0,1) -- (1,0); \\end{tikzpicture}",
                "wrong_return_value",
            ]
        )

        # expected result
        expected = {
            "individual_scores": {"example1": True, "example2": False},
            "varscore": 0.5,
        }

        self.assertEqual(evaluate(dummy_dataset, model), expected)

    def test_evaluator_metric_exists_complex(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1", "example2"],
                "code": [
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}",
                    "\\begin{tikzpicture} \\draw (0,0) -- (1,1); \\end{tikzpicture}",
                ],
                "instruction": ["Rotate the line", "Rotate the line"],
                "diffs": [
                    ["abc", "def", "@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0"],
                    ["abc", "@@ -30 +30 @@\n-0+1@@ -39 +39 @@\n-1+0", "def"],
                ],
            }
        )

        # mock llm_model
        model: LLM_Model = LLM_Model()
        model.batchRequest = MagicMock(
            return_value=[
                "\\begin{tikzpicture} \\draw (0,1) -- (1,0); \\end{tikzpicture}",
                "\\begin{tikzpicture} \\draw (0,1) -- (1,0); \\end{tikzpicture}",
            ]
        )

        # expected result
        expected = {
            "individual_scores": {"example1": True, "example2": True},
            "varscore": 1.0,
        }

        self.assertEqual(evaluate(dummy_dataset, model), expected)


if __name__ == "__main__":
    unittest.main()
