import unittest
import os
import timeout_decorator
from varbench.compilers import Compiler, TexCompiler
from varbench.evaluation.evaluator import evaluate
from varbench.model import LLM_Model
from unittest.mock import MagicMock
from datasets import Dataset
from PIL import Image
import difflib
import re

#@@ -7 +7 @@\n-\begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture}+\begin{tikzpicture} \draw (0,1) -- (1,0); \end{tikzpicture}


class TestEvaluator(unittest.TestCase):

    def setUp(self) -> None:
        def _diffs(input, predictions: list) -> list:
            predictions = [prediction.split("\n") for prediction in predictions]
            return [
                "".join(list(difflib.unified_diff(input.split("\n"), prediction, n=0))[2:])
                for prediction in predictions
            ]
        def _get_first_code_block(text):
            # Regular expression to find the first code block, ignoring the language specifier
            match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
            return match.group(1).strip() if match else None
        with open("tests/resources/tikz/input.md") as in_text:
            self.input_tex = _get_first_code_block(in_text.read())
        with open("tests/resources/tikz/reference.md") as in_text:
            self.ref_tex = in_text.read()
        self.ref_diff = _diffs(self.input_tex,[_get_first_code_block(self.ref_tex)])[0]
        print(self.ref_diff)
        self.compiler: TexCompiler = TexCompiler()
        self.compiler.compile_from_string = MagicMock(
            return_value=Image.open("tests/resources/images/reference.jpeg")
        )

        self.model: LLM_Model = LLM_Model("model-name", 0)

        return super().setUp()
    def test_evaluator_metric_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [self.input_tex],
                "instruction": ["Rotate the line"],
                "diffs": [[self.ref_diff]],
                "result_description": [
                    "a line going from the top left to the bottom right"
                ],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg")],
            }
        )

        # mock llm_model
        self.model.batchRequest = MagicMock(return_value=[[self.ref_tex]])

        # expected result
        expected = {"diffs_score": 1.0}
        actual = evaluate(dummy_dataset, self.model, self.compiler)
        self.assertEqual(
            actual[0].get("diffs_score"),
            expected["diffs_score"],
        )

    def test_evaluator_metric_not_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [self.input_tex],
                "instruction": ["Rotate the line"],
                "diffs": [[self.ref_diff]],
                "result_description": [
                    "a line going from the top left to the bottom right"
                ],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg")],
            }
        )

        # mock llm_model
        self.model.batchRequest = MagicMock(return_value=[["wrong_return_value"]])

        # expected result
        expected = {"diffs_score": 0.0}

        self.assertEqual(
            evaluate(dummy_dataset, self.model, self.compiler)[0].get("diffs_score"),
            expected["diffs_score"],
        )

    def test_evaluator_metric_exists_multiple(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1", "example2"],
                "code": [
                    self.input_tex,
                    self.input_tex,
                ],
                "instruction": ["Rotate the line", "Rotate the line"],
                "diffs": [
                    [self.ref_diff],
                    [self.ref_diff],
                ],
                "result_description": [
                    "a line going from the top left to the bottom right",
                    "a line going from the top left to the bottom right"
                ],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg"),Image.open("tests/resources/images/reference.jpeg")],
            }
        )

        # mock llm_model
        self.model.batchRequest = MagicMock(
            return_value=[
                [self.ref_tex],
                ["wrong_return_value"],
            ]
        )

        # expected result
        expected = {
            "diffs_score": 0.5,
        }

        self.assertEqual(
            evaluate(dummy_dataset, self.model, self.compiler)[0].get("diffs_score"),
            expected["diffs_score"],
        )

    def test_evaluator_metric_exists_complex(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1", "example2"],
                "code": [
                    self.input_tex,
                    self.input_tex,
                ],
                "instruction": ["Rotate the line", "Rotate the line"],
                "diffs": [
                    ["abc", "def", self.ref_diff],
                    ["abc", self.ref_diff, "def"],
                ],
                "result_description": [
                    "a line going from the top left to the bottom right",
                    "a line going from the top left to the bottom right"
                ],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg"),Image.open("tests/resources/images/reference.jpeg")],
            }
        )

        # mock llm_model
        self.model.batchRequest = MagicMock(
            return_value=[
                [self.ref_tex],
                [self.ref_tex],
            ]
        )

        # expected result
        expected = {
            "diffs_score": 1.0,
        }

        self.assertEqual(
            evaluate(dummy_dataset, self.model, self.compiler)[0].get("diffs_score"),
            expected["diffs_score"],
        )


if __name__ == "__main__":
    unittest.main()
