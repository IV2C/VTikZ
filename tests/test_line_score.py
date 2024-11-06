import unittest
from varbench.evaluation.line_diff_scorer import compute_line_score


class TestLineScore(unittest.TestCase):

    def diff_from_number(self, number: list[int]) -> str:
        return "\n".join([f"@@ -{n} +{n} @@" for n in number])

    def test_line_score_simple(self):
        predictions = [[self.diff_from_number([5, 10, 15])]]
        references = [[self.diff_from_number([5, 10, 15])]]
        expected = [1.0]

        self.assertEqual(compute_line_score(predictions, references), expected)

    def test_line_score_max_single(self):
        predictions = [[self.diff_from_number([5, 10, 15])]]
        references = [
            [self.diff_from_number([5, 10, 15]), self.diff_from_number([5, 20, 25])]
        ]
        expected = [1.0]

        self.assertEqual(compute_line_score(predictions, references), expected)

    def test_line_score_not_max(self):
        predictions = [[self.diff_from_number([5, 10, 15])]]
        references = [[self.diff_from_number([5, 10, 20])]]
        expected = [2 / 3]
        self.assertAlmostEqual(
            compute_line_score(predictions, references), expected, delta=0.01
        )

    def test_line_score_not_max_two(self):
        predictions = [[self.diff_from_number([5, 10, 15])]]
        references = [
            [self.diff_from_number([5, 10, 20]), self.diff_from_number([5, 10, 20])]
        ]
        expected = [2 / 3]
        self.assertAlmostEqual(
            compute_line_score(predictions, references), expected, delta=0.01
        )

    def test_line_score_max_two_two(self):
        predictions = [
            [self.diff_from_number([5, 10, 15]), self.diff_from_number([5, 20, 15])]
        ]
        references = [
            [self.diff_from_number([5, 10, 20]), self.diff_from_number([5, 20, 15])]
        ]
        expected = [1]
        self.assertAlmostEqual(
            compute_line_score(predictions, references), expected, delta=0.01
        )

    def test_line_score_not_max_two_two(self):
        predictions = [
            [
                self.diff_from_number([5, 10, 15, 22]),
                self.diff_from_number([5, 20, 15, 30]),
            ]
        ]
        references = [
            [
                self.diff_from_number([5, 10, 12, 25]),
                self.diff_from_number([5, 20, 15, 25]),
            ]
        ]
        expected = [3/4]
        self.assertAlmostEqual(
            compute_line_score(predictions, references), expected, delta=0.01
        )

    def test_line_score_not_max_max_two_two(self):
        predictions = [
            [
                self.diff_from_number([5, 10, 15, 22]),
                self.diff_from_number([5, 20, 15, 30]),
            ],
                     [
                self.diff_from_number([5, 10, 15, 22]),
                self.diff_from_number([5, 20, 15, 30]),
            ]
        ]
        references = [
            [
                self.diff_from_number([5, 10, 12, 25]),
                self.diff_from_number([5, 20, 15, 25]),
            ],
            [
                self.diff_from_number([5, 10, 12, 25]),
                self.diff_from_number([5, 20, 15, 30]),
            ]
        ]
        expected = [3/4,1]
        self.assertAlmostEqual(
            compute_line_score(predictions, references), expected, delta=0.01
        )
