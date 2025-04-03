from varbench.agents.agent import Agent
from typing import Iterable
from PIL import Image
from vif_agent.agent import VifAgent
from varbench.agents.external.external_agent import ExternalAgent
from renderers import TexRenderer
from varbench.utils.parsing import get_config


class VIFAgent(ExternalAgent):
    def __init__(self, **kwargs):
        renderer = TexRenderer()  # hardcoded for now
        vif_args = {**get_config("VIF")}
        self.agent = VifAgent(code_renderer=renderer.from_string_to_image, **vif_args)
        super().__init__(**kwargs)

    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:

        return self.agent.apply_instruction(code, instruction)

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
