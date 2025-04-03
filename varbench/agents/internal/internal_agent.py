


from typing import Iterable
from varbench.agents import Agent
from varbench.api.chat_api import ChatApi
from PIL import Image

class InternalAgent(Agent):
    def __init__(self, api: ChatApi, **kwargs) -> None:
        self.api = api
        pass

