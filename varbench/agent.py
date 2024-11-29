from abc import ABC, abstractmethod
import base64
from io import BytesIO
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

from varbench.api.chat_api import ChatApi
from varbench.prompt_templates import (
    IT_PROMPT,
    MULTIMODAL_INSTRUCTION,
    MULTIMODAL_LOOP_INSTRUCTION,
    SYSTEM_PROMPT_GENERATION,
    SYSTEM_PROMPT_GENERATION_VLM_LOOP,
    VLM_INSTRUCTION,
)
from PIL import Image

from varbench.renderers.renderer import Renderer
from varbench.utils.parsing import get_first_code_block
from loguru import logger


class Agent:
    def __init__(self, api: ChatApi, **kwargs) -> None:
        self.api = api
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


class SimpleLLMAgent(Agent):
    """Simple LLM agent that uses only the "reading" capabilities of the models tested"""

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:
        messages = self._create_message(instruction, code)

        return self.api.chat_request(messages)

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        image_input: Iterable[Image.Image] = None,
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


class LMMAgent(Agent):
    """LMM(Large Multimodal Model) agent that uses both the "reading" and "vision" capabilities of the models tested"""

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:
        messages = self._create_message(instruction, code, image)

        return self.api.chat_request(messages)

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        images_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        messages = [
            self._create_message(instruction, code, image)
            for instruction, code, image in zip(instructions, codes, images_input)
        ]

        return self.api.batch_chat_request(messages, ids)

    def _create_message(
        self, instruction: str, code: str, image: Image.Image
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""

        # Some multimodal models do not support system prompts, so we only use the user prompt as input

        user_instruction = MULTIMODAL_INSTRUCTION.format(
            instruction=instruction, content=code
        )

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        return messages


class VLLMLoopAgent(Agent):
    """A LLM and a VLM agent interacting until the VLM is satisfied"""

    def __init__(
        self,
        api: ChatApi,
        vlm_api: ChatApi,
        renderer: Renderer,
        interaction_nb=3,
        **kwargs,
    ) -> None:
        self.renderer = renderer
        self.vlm_api = vlm_api
        self.interation_nb = interaction_nb
        super().__init__(api, **kwargs)

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:

        # getting the n and overriding it to handle single requests manually
        request_n = self.api.n
        self.api.n = 1
        results = [self.single_loop(instruction, code) for _ in range(self.api.n)]
        self.api.n = request_n
        return results

    def single_loop(self, instruction: str, code: str) -> str:
        # generating results
        messages: list = self._create_message_llm(instruction, code)
        response = self.api.chat_request(messages)[0]
        code_response = get_first_code_block(response)

        # making m VLM/LLM interactions
        for i in range(self.interation_nb):
            # adding the code response to the conversation
            messages.append({"role": "assistant", "content": response})
            computed_image = self.renderer.from_string_to_image(code_response)
            vlm_message = self._create_message_vlm(instruction, computed_image)
            vlm_remark = self.vlm_api.chat_request(vlm_message)[0]
            messages.append({"role": "user", "content": vlm_remark})
            response = self.api.chat_request(messages)[0]
            code_response = get_first_code_block(response)
        messages.append({"role": "assistant", "content": response})
        logger.info(messages)
        return response

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        images_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [
            self.compute(instruction, code, image)
            for instruction, code, image in zip(instructions, codes, images_input)
        ]

    def _create_message_llm(
        self, instruction: str, code: str
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""
        user_instruction = IT_PROMPT.format(instruction=instruction, content=code)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_GENERATION_VLM_LOOP,
            },
            {"role": "user", "content": user_instruction},
        ]

        return messages

    def _create_message_vlm(
        self, instruction: str, image: Image.Image
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""

        # Some multimodal models do not support system prompts, so we only use the user prompt as input

        user_instruction = VLM_INSTRUCTION.format(instruction=instruction)

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        return messages


class LMMLoopAgent(Agent):
    """A LLM and a VLM agent interacting until the VLM is satisfied"""

    def __init__(
        self,
        api: ChatApi,
        renderer: Renderer,
        interaction_nb=3,
        **kwargs,
    ) -> None:
        self.renderer = renderer
        self.interation_nb = interaction_nb
        super().__init__(api, **kwargs)

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:

        # getting the n and overriding it to handle single requests manually
        request_n = self.api.n
        self.api.n = 1
        results = [self.single_loop(instruction, code) for _ in range(self.api.n)]
        self.api.n = request_n
        return results

    def single_loop(self, instruction: str, code: str) -> str:
        # generating results
        messages: list = self._create_message_llm(instruction, code)
        response = self.api.chat_request(messages)[0]
        code_response = get_first_code_block(response)

        # making self-refining interactions
        for _ in range(self.interation_nb):
            # adding the code response to the conversation
            messages.append({"role": "assistant", "content": response})
            computed_image = self.renderer.from_string_to_image(code_response)
            user_message = self._create_self_revise_message_multimodal(computed_image)
            messages.append({"role": "user", "content": user_message})
            response = self.api.chat_request(messages)[0]
            
            if "instruction satified" in response:
                break
            code_response = get_first_code_block(response)
        messages.append({"role": "assistant", "content": response})
        logger.info(messages)
        return response

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        images_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [
            self.compute(instruction, code, image)
            for instruction, code, image in zip(instructions, codes, images_input)
        ]

    def _create_message(
        self, instruction: str, code: str, image: Image.Image
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""

        # Some multimodal models do not support system prompts, so we only use the user prompt as input

        user_instruction = IT_PROMPT.format(
            instruction=instruction, content=code
        )

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        return messages

    def _create_self_revise_message_multimodal(
        self, image: Image.Image
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_GENERATION,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": MULTIMODAL_LOOP_INSTRUCTION,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        return messages


def instantiate_agent(
    agent_str: str,
    api: ChatApi,
    vlm_api: ChatApi = None,
    renderer: Renderer = None,
    interaction_nb=3,
) -> Agent:
    agent_map = {
        "simpleLLM": SimpleLLMAgent,
        "simpleLMM": LMMAgent,
        "loopVLMLLM": VLLMLoopAgent,
    }

    key_args = {}
    key_args["api"] = api
    key_args["vlm_api"] = vlm_api
    key_args["renderer"] = renderer
    key_args["interaction_nb"] = interaction_nb

    return agent_map[agent_str](**key_args)
