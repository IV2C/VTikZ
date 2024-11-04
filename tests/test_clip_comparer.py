import unittest

from varbench.evaluation.clip_comparer import ClipComparer
from PIL import Image


class TestClipComparer(unittest.TestCase):

    def test_clip_score(self):
        clip_comparer: ClipComparer = ClipComparer(force_cpu=True)
        images = [
            [
                Image.open("tests/resources/images/dog-redeye.jpeg"),
                Image.open("tests/resources/images/dog.jpeg"),
                Image.open("tests/resources/images/dog-top-left.jpeg"),
                Image.open("tests/resources/images/dog-rotated.jpeg"),
            ]
        ]
        result_descriptions = ["a drawing of a dog with red eyes"]
        results = clip_comparer.clip_scores(images, result_descriptions)
        self.assertTrue(results[0][0] == max(results[0]))

    def test_image_comparison(self):
        clip_comparer: ClipComparer = ClipComparer(force_cpu=True)
        images = [
            [
                Image.open("tests/resources/images/dog-redeye.jpeg"),
                Image.open("tests/resources/images/dog.jpeg"),
                Image.open("tests/resources/images/dog-top-left.jpeg"),
                Image.open("tests/resources/images/dog-rotated.jpeg"),
            ]
        ]
        ref_images = [Image.open("tests/resources/images/dog-redeye.jpeg")]
        results = clip_comparer.image_similarities(images, ref_images)
        self.assertTrue(results[0][0] == max(results[0]))


if __name__ == "__main__":
    unittest.main()
