import base64
from io import BytesIO
from typing import Iterable
from PIL import Image
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from varbench.agents import Agent
from varbench.api.chat_api import ChatApi
from varbench.prompt_templates import IT_PROMPT, SYSTEM_PROMPT_GENERATION_VLM_LOOP, VLM_INSTRUCTION
from varbench.renderers.renderer import Renderer, RendererException
from varbench.utils.parsing import get_first_code_block

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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]

        return messages
