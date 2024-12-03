import unittest
import os
import timeout_decorator
from varbench.api.chat_api import ChatApi, VLLMApi
from varbench.evaluation.metrics import Metric, PatchMetric
from varbench.renderers import Renderer, TexRenderer
from varbench.evaluation.evaluator import evaluate
from varbench.agents import Agent
from unittest.mock import MagicMock
from datasets import Dataset
from PIL import Image
import difflib
import re

from varbench.utils.parsing import get_first_code_block

# @@ -7 +7 @@\n-\begin{tikzpicture} \draw (0,0) -- (1,1); \end{tikzpicture}+\begin{tikzpicture} \draw (0,1) -- (1,0); \end{tikzpicture}


class TestEvaluator(unittest.TestCase):

    def setUp(self) -> None:
        def _patches(input, predictions: list) -> list:
            predictions = [prediction.split("\n") for prediction in predictions]
            return [
                "".join(
                    list(difflib.unified_diff(input.split("\n"), prediction, n=0))[2:]
                )
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
        self.ref_patch = _patches(
            self.input_tex, [_get_first_code_block(self.ref_tex)]
        )[0]
        self.wrong_patch = "@@ -375 +375 @@"
        print(self.ref_patch)
        self.dummyMetric: Metric = [PatchMetric()]

        return super().setUp()

    def test_evaluator_metric_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [self.input_tex],
                "instruction": ["Rotate the line"],
                "patch": [self.ref_patch],
                "result_description": [
                    "a line going from the top left to the bottom right"
                ],
                "predictions":[[get_first_code_block(self.ref_tex)]],
                "images_result": [[Image.open("tests/resources/images/reference.jpeg")]],
                "image_input": [Image.open("tests/resources/images/reference.jpeg")],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg")],
            }
        )

        # expected result
        expected = {"PatchMetric": 100.0}
        actual = evaluate(dummy_dataset, self.dummyMetric)
        self.assertEqual(
            actual[0].get("PatchMetric"),
            expected["PatchMetric"],
        )

    def test_evaluator_metric_not_exists(self):

        # dataset
        dummy_dataset: Dataset = Dataset.from_dict(
            {
                "id": ["example1"],
                "code": [self.input_tex],
                "instruction": ["Rotate the line"],
                "patch": [self.ref_patch],
                "result_description": [
                    "a line going from the top left to the bottom right"
                ],
                
                "predictions":[["wrong patch"]],
                "images_result": [[Image.open("tests/resources/images/reference.jpeg")]],
                "image_input": [Image.open("tests/resources/images/reference.jpeg")],
                "image_solution": [Image.open("tests/resources/images/reference.jpeg")],
            }
        )


        # expected result
        expected = {"PatchMetric": 0.0}

        self.assertEqual(
            evaluate(dummy_dataset, self.dummyMetric)[0].get(
                "PatchMetric"
            ),
            expected["PatchMetric"],
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
                "patch": [
                    self.ref_patch,
                    self.ref_patch,
                ],
                "result_description": [
                    "a line going from the top left to the bottom right",
                    "a line going from the top left to the bottom right",
                ],
                "image_solution": [
                    Image.open("tests/resources/images/reference.jpeg"),
                    Image.open("tests/resources/images/reference.jpeg"),
                ],
                "image_input": [
                    Image.open("tests/resources/images/reference.jpeg"),
                    Image.open("tests/resources/images/reference.jpeg"),
                ],
                
                "predictions":[[get_first_code_block(self.ref_tex)],["wrong patch"]],
                "images_result": [[Image.open("tests/resources/images/reference.jpeg")],[Image.open("tests/resources/images/reference.jpeg")]],
            }
        )



        # expected result
        expected = {
            "PatchMetric": 50,
        }

        self.assertEqual(
            evaluate(dummy_dataset, self.dummyMetric)[0].get(
                "PatchMetric"
            ),
            expected["PatchMetric"],
        )


if __name__ == "__main__":
    unittest.main()
