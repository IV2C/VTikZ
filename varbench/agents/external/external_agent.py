from varbench.agents.agent import Agent
from typing import Iterable
from PIL import Image
from vif_agent.agent import VifAgent  

class ExternalAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
