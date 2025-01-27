from varbench.api.chat_api import ChatApi
from varbench.utils.prompts.simple_templates import (
    CODE_CORRECTOR_IT_PROMPT,
    CODE_CORRECTOR_SYSTEM_PROMPT,
)
from varbench.renderers import Renderer, RendererException
from varbench.utils.parsing import get_config, get_first_code_block, replace_first_code_block

from loguru import logger
from PIL import Image


class CodeCorrectAgent:
    def __init__(
        self, renderer: Renderer, api: ChatApi, max_iteration: int = 0
    ) -> None:
        """instantiates a code_correct agent

        Args:
            renderer (Renderer): the renderer to use
            max_iteration (int): maximum number of iterations to use, note: if set to 0 the code correct agent won't be used at all
        """

        llm_args = {**get_config("CODE_CORRECT_AGENT")}
        self.renderer = renderer
        self.max_iteration = llm_args["max_iteration"] or max_iteration
        self.api: ChatApi = api
        if max_iteration > 0:
            logger.info(f"code correct agent instantiated {self}")
        pass

    def get_correct_code(self, code: str) -> tuple[Image.Image, str]:
        """Tries to render an image from the code in the text given, self-iterates with the agent in case of an error

        Args:
            code (str): the message containing the code to render
        Returns:
            tuple[Image.Image,str]: The image rendered and the associated response(the original one with the code replace with a working one)
        """
        original_response = code
        code = get_first_code_block(code)
        if len(code.split("\n")) < 5:
            logger.warning(f"The following code is too short to be corrected \n {code}")
            return None, None
        for _ in range(self.max_iteration):
            try:
                return self.renderer.from_string_to_image(code), replace_first_code_block(original_response,code)
            except RendererException as re:
                logger.info("Rendering code failed, CodeCorrect Agent trying to correct...")
                
                messages = [{"role": "system", "content": CODE_CORRECTOR_SYSTEM_PROMPT}]
                user_message = CODE_CORRECTOR_IT_PROMPT.format(
                    error_message=re.extract_error(), code=code
                )
                messages.append({"role": "user", "content": user_message})
                correct_agent_response = self.api.chat_request(messages)[0]
                logger.warning(correct_agent_response)
                code = get_first_code_block(correct_agent_response)
        try:
            return self.renderer.from_string_to_image(code), replace_first_code_block(original_response,code)
        except RendererException as re:
            return None, None

    def __str__(self) -> str:
        return (
            f"CodeCorrectAgent(renderer={self.renderer}, "
            f"max_iteration={self.max_iteration}, api={self.api})"
        )
