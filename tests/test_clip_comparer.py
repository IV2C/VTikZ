import unittest

from vtikz.evaluation.clip_comparer import ClipComparer
from PIL import Image
import os

class TestClipComparer(unittest.TestCase):

    @unittest.skipIf(os.environ.get("CI"), "Too much storage needed for running in CI")
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
        results = clip_comparer.text_similarities(images, result_descriptions)
        self.assertTrue(results[0] == max(results))
        
    @unittest.skipIf(os.environ.get("CI"), "Too much storage needed for running in CI")
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
        self.assertTrue(results[0] == max(results))


if __name__ == "__main__":
    unittest.main()
