import base64
from io import BytesIO
from typing import Iterable
from PIL import Image
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from varbench.agents import Agent, CodeCorrectAgent
from varbench.api.chat_api import ChatApi
from varbench.renderers.renderer import Renderer
from varbench.utils.parsing import get_first_code_block
from varbench.prompt_templates import (
    IT_PROMPT,
    MULTIMODAL_LOOP_INSTRUCTION,
    MULTIMODAL_LOOP_SYSTEM_INSTRUCTION,
)


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
        self.debug_store_conv = kwargs["debug_store_conv"]
        self.correct_code_agent: CodeCorrectAgent = CodeCorrectAgent(self.renderer, api)
        super().__init__(api, **kwargs)

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:

        # getting the n and overriding it to handle single requests manually
        request_n = self.api.n
        self.api.n = 1
        results = [
            self.single_loop(instruction, code, image) for _ in range(self.api.n)
        ]
        self.api.n = request_n
        return results

    def single_loop(self, instruction: str, code: str, image: Image.Image) -> str:
        # generating results
        messages: list = self._create_message(instruction, code, image)
        original_response = self.api.chat_request(messages)[0]
        computed_image, original_response = self.correct_code_agent.get_correct_code(
            original_response
        )
        messages.append({"role": "assistant", "content": original_response})

        # making self-refining interactions
        for i in range(self.interation_nb):

            user_message = self._create_self_revise_message_multimodal(computed_image)
            messages.extend(user_message)
            response = self.api.chat_request(messages)[0]

            if (
                "instruction satisfied" in response or "```" not in response
            ):  # stopping the conversation if the instruction is satisfied
                break
            # calling the code correcting agent
            computed_image, response = self.correct_code_agent.get_correct_code(
                response
            )

            # adding the code response to the conversation
            messages.append({"role": "assistant", "content": response})

            if not computed_image or not response:
                return (
                    None  # the code correcting agent did not manage to correct the code
                )

        logger.info(messages)
        return get_first_code_block(response)

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

        user_instruction = IT_PROMPT.format(instruction=instruction, content=code)

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": MULTIMODAL_LOOP_SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]

        return messages

    def _create_self_revise_message_multimodal(
        self, image: Image.Image
    ) -> Iterable[ChatCompletionMessageParam]:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": MULTIMODAL_LOOP_INSTRUCTION,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]

        return messages
