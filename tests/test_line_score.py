import unittest
from varbench.evaluation.line_patch_scorer import compute_line_score


class TestLineScore(unittest.TestCase):

    def patch_from_number(self, number: list[int]) -> str:
        return "\n".join([f"@@ -{n} +{n} @@" for n in number])

    def test_line_score_simple(self):
        predictions = [[self.patch_from_number([5, 10, 15])]]
        references = [self.patch_from_number([5, 10, 15])]
        expected = [[100.0]]

        self.assertEqual(compute_line_score(predictions, references), expected)

    def test_line_score_not_max(self):
        predictions = [[self.patch_from_number([5, 10, 15])]]
        references = [self.patch_from_number([5, 10, 20])]
        full_expected = [[200 / 3]]
        full_computed = compute_line_score(predictions, references)
        for expected,computed in zip(full_expected,full_computed):
            for single_expected,single_computed in zip(expected,computed):
                self.assertAlmostEqual(single_computed, single_expected)

    def test_line_score_max_two_two(self):
        predictions = [
            [self.patch_from_number([5, 10, 15]), self.patch_from_number([5, 20, 10])]
        ]
        references = [self.patch_from_number([5, 10, 20])]
        full_expected = [[200 / 3, 100]]
        full_computed = compute_line_score(predictions, references)
        
        for expected,computed in zip(full_expected,full_computed):
            for single_expected,single_computed in zip(expected,computed):
                self.assertAlmostEqual(single_computed, single_expected)

    def test_line_score_not_max_two_two(self):
        predictions = [
            [
                self.patch_from_number([5, 10, 15, 22]),
                self.patch_from_number([5, 20, 15, 30]),
            ]
        ]
        references = [self.patch_from_number([5, 10, 12, 25])]
        full_expected = [[200 / 4, 100 / 4]]
        full_computed = compute_line_score(predictions, references)
        for expected,computed in zip(full_expected,full_computed):
            for single_expected,single_computed in zip(expected,computed):
                self.assertAlmostEqual(single_computed, single_expected)
    def test_line_score_not_max_max_two_two(self):
        predictions = [
            [
                self.patch_from_number([5, 10, 15, 22]),
                self.patch_from_number([5, 20, 15, 30]),
            ],
            [
                self.patch_from_number([5, 10, 15, 22]),
                self.patch_from_number([5, 20, 15, 30]),
            ],
        ]
        references = [
            self.patch_from_number([5, 20, 15, 25]),
            self.patch_from_number([5, 20, 15, 30]),
        ]
        full_expected = [[200 / 4, 300 / 4], [200 / 4, 100.0]]
        full_computed = compute_line_score(predictions, references)
        for expected,computed in zip(full_expected,full_computed):
            for single_expected,single_computed in zip(expected,computed):
                self.assertAlmostEqual(single_computed, single_expected)
