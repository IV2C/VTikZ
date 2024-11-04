import torch
import open_clip

from PIL import Image
from typing import Callable


class ClipComparer:
    """
    Class to compute similarity scores between images and text descriptions or between pairs of images
    using a CLIP (Contrastive Language-Image Pretraining) model.
    """

    def __init__(
        self,
        model_name: str = "ViT-bigG-14-quickgelu",
        pretrained_name: str = "metaclip_fullcc",
        force_cpu: bool = False,
        policy: Callable[[list[float]], float] = lambda l: max(l) if len(l)>0 else 0,
    ) -> None:
        """
        Initializes the ClipComparer with a specified model and pretrained weights.

        Args:
            model_name (str): The name of the CLIP model variant. Defaults to "ViT-bigG-14-quickgelu".
            pretrained_name (str): The pretrained weights to load. Defaults to "metaclip_fullcc".
            policy (Callable[[list[float]],float]): policy for the computation of score from the pass@
        """
        device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_name, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.policy = policy
        self.model.eval()

    def clip_scores(
        self, images: list[list[Image.Image]], result_descriptions: list[str]
    ) -> list[float]:
        """
        Calculates similarity scores between images and corresponding text descriptions.

        Args:
            images (list[list[Image.Image]]): A list of lists of images to compare.
            result_descriptions (list[str]): A list of text descriptions for each list of images.

        Returns:
            list[float]:  A list of similarity scores, containing
                               scores between each image in `images` and its corresponding description, computed according to the policy.
        """
        results: list[list[float]] = []

        with torch.no_grad():
            for image_list, result_description in zip(images, result_descriptions):
                # Encode the text and normalize features
                text_features = self.model.encode_text(
                    self.tokenizer(result_description)
                )
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Encode and normalize image features
                images_features = self._encode_images(image_list)

                cos_similarities = [
                    (100.0 * image_feature @ text_features.T).tolist()[0][0]
                    for image_feature in images_features
                ]
                results.append(self.policy(cos_similarities))
        return results

    def image_similarities(
        self, images: list[list[Image.Image]], references: list[Image.Image]
    ) -> list[float]:
        """
        Calculates similarity scores between images and a reference image.

        Args:
            images (list[list[Image.Image]]): A list of lists of images to compare.
            references (list[Image.Image]): A list of reference images.

        Returns:
            list[float]: A list of similarity scores, containing
                               scores between each image in `images` and its corresponding reference image, computed according to the policy.
        """
        results: list[list[float]] = []

        with torch.no_grad():
            for image_list, reference in zip(images, references):
                # Encode reference image and normalize features
                ref_features = self._encode_images([reference])[0]

                # Encode and normalize image features
                images_features = self._encode_images(image_list)

                cos_similarities = [
                    (100.0 * image_feature @ ref_features.T).tolist()[0][0]
                    for image_feature in images_features
                ]
                results.append(self.policy(cos_similarities))
        return results

    def _encode_images(self, image_list: list[Image.Image]) -> list[torch.Tensor]:
        """
        Helper method to encode and normalize a list of images.

        Args:
            image_list (list[Image.Image]): A list of images to process.

        Returns:
            list[torch.Tensor]: A list of normalized feature tensors for each image.
        """
        processed_images = [self.preprocess(image).unsqueeze(0) for image in image_list]
        images_features = [
            self.model.encode_image(processed_image)
            for processed_image in processed_images
        ]
        images_features = [
            image_feature / image_feature.norm(dim=-1, keepdim=True)
            for image_feature in images_features
        ]
        return images_features
