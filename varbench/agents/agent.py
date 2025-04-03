


from typing import Iterable
from varbench.api.chat_api import ChatApi
from PIL import Image

class Agent:
    def __init__(self, **kwargs) -> None:
        pass

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:
        """
        Computes a response based on the given instruction and code.

        Args:
            instruction (str): The instruction or prompt provided to guide the computation.
            code (str): The code or content to process based on the instruction.
            **kwargs: Additional optional parameters for flexibility in extended implementations.

        Returns:
            list[str]: A list of strings representing the computed responses from the API.
        """
        pass

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        image_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        """
        Computes responses for multiple instruction-code pairs, ensuring output order aligns with provided IDs.

        Args:
            instructions (Iterable[str]): A collection of instructions for batch processing.
            codes (Iterable[str]): Corresponding code snippets or content to process with the instructions.
            ids (Iterable[str]): Unique identifiers to ensure correct alignment of results in the output.
            image_input (Iterable[str]): The image computed from the input code.
            **kwargs: Additional optional parameters for extended agent configurations.

        Returns:
            Iterable[list[str]]: An iterable of lists, each containing computed responses for the respective input pair.

        Notes:
            - The implementation supports batch processing for APIs that allow it, ensuring ID-matching order.
            - Input collections (`instructions`, `codes`, `ids`) must have matching lengths.
        """
        pass

