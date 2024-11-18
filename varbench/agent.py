from abc import ABC, abstractmethod
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

from varbench.api.chat_api import ChatApi
from varbench.prompt_templates import IT_PROMPT, SYSTEM_PROMPT_GENERATION


class Agent:
    def __init__(self, api: ChatApi, **kwargs) -> None:
        self.api = api
        pass

    def compute(self, instruction: str, code: str, **kwargs) -> Iterable[str]:
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
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        """
        Computes responses for multiple instruction-code pairs, ensuring output order aligns with provided IDs.

        Args:
            instructions (Iterable[str]): A collection of instructions for batch processing.
            codes (Iterable[str]): Corresponding code snippets or content to process with the instructions.
            ids (Iterable[str]): Unique identifiers to ensure correct alignment of results in the output.
            **kwargs: Additional optional parameters for extended agent configurations.

        Returns:
            Iterable[list[str]]: An iterable of lists, each containing computed responses for the respective input pair.

        Notes:
            - The implementation supports batch processing for APIs that allow it, ensuring ID-matching order.
            - Input collections (`instructions`, `codes`, `ids`) must have matching lengths.
        """
        pass


class SimpleLLMAgent(Agent):
    """Simple LLM agent that uses only the "reading" capabilities of the models tested

    Args:
        Agent (_type_): _description_
    """

    def compute(self, instruction: str, code: str, **kwargs) -> Iterable[str]:
        messages = self._create_message(instruction, code)

        return self.api.chat_request(messages)

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        messages = [
            self._create_message(instruction, code)
            for instruction, code in zip(instructions, codes)
        ]

        return self.api.batch_chat_request(messages, ids)

    def _create_message(
        self, instruction: str, code: str
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""
        user_instruction = IT_PROMPT.format(instruction=instruction, content=code)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_GENERATION,
            },
            {"role": "user", "content": user_instruction},
        ]

        return messages
