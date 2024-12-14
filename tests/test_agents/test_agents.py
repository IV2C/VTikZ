import os
import unittest
from PIL import Image

from varbench.agents import Agent, LMMLoopAgent
from varbench.api.chat_api import ChatApi, OpenAIApi
from varbench.renderers.tex_renderer import TexRenderer
from varbench.utils.parsing import get_first_code_block, replace_first_code_block


@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestAgentLMMLoopOpenAI(unittest.TestCase):
    def setUp(self) -> None:
        self.instruction = "Add a simple crown on the top of the squid's face"
        with open("tests/resources/tikz/squid.tex") as in_text:
            self.squid_tikz = in_text.read()
        self.renderer = TexRenderer()
        self.squid_image: Image.Image = self.renderer.from_string_to_image(
            self.squid_tikz
        )

    def test_multimodal_loop_agent_openai(self):
        chat_api: ChatApi = OpenAIApi(1, 1, "gpt-4o")

        multimodal_openai_agent: Agent = LMMLoopAgent(
            chat_api, self.renderer, 3, debug_store_conv=True
        )

        response = multimodal_openai_agent.compute(
            self.instruction, self.squid_tikz, self.squid_image
        )
        print(response)

        