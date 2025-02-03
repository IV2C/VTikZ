import base64
from io import BytesIO
from typing import Iterable
from PIL import Image
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from varbench.agents import Agent
from varbench.utils.prompts.simple_templates import IT_PROMPT
from varbench.utils.prompts.FAR_template import FAR_SYSTEM_PROMPT
from varbench.utils.parsing import apply_far_edit, get_first_code_block


class FARAgent(Agent):
    """Simple LLM agent that uses only the "reading" capabilities of the models tested,
    instead of sending the full code back, this agent
    only returns a list of blocks to replace, replaced with a find-and-replace function
    """

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:
        messages = self._create_message(instruction, code)
        all_edits = self.api.chat_request(messages)
        parsed_all_edits = [get_first_code_block(edits) for edits in all_edits]
        all_edited_codes = [
            apply_far_edit(code, parsed_edits) for parsed_edits in parsed_all_edits
        ]
        return [
            edits.replace("`", "") + "```tikz\n" + edited_response + "\n```\n"
            for edited_response, edits in zip(all_edited_codes, all_edits)
        ]  # we return the full thought process of the LLM, by adding back its original response in the message

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        image_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [
            self.compute(instruction, code)
            for instruction, code in zip(instructions, codes)
        ]

    def _create_message(
        self, instruction: str, code: str
    ) -> Iterable[ChatCompletionMessageParam]:
        """Add a row that contains the prompt"""
        user_instruction = IT_PROMPT.format(instruction=instruction, content=code)

        messages = [
            {
                "role": "system",
                "content": FAR_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_instruction},
        ]

        return messages
