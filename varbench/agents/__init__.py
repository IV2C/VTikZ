from .agent import Agent
from .internal.code_Correct_Agent import CodeCorrectAgent
from .internal.LMM_loop_agent import LMMLoopAgent
from .internal.simple_LLM_agent import SimpleLLMAgent
from .internal.LMM_agent import LMMAgent
from .internal.VLLM_loop_agent import VLLMLoopAgent
from .internal.FAR_LLM_Agent import FARAgent

from .external.VIF_agent import VIFAgent

def instantiate_agent(
    agent_str: str,
    api=None,
    vlm_api=None,
    renderer=None,
    interaction_nb=3,
) -> Agent:
    agent_map = {
        "simpleLLM": SimpleLLMAgent,
        "simpleLMM": LMMAgent,
        "loopVLMLLM": VLLMLoopAgent,
        "loopLMM": LMMLoopAgent,
        "FAR": FARAgent,
        "VIF":VIFAgent
    }

    key_args = {}
    key_args["api"] = api
    key_args["vlm_api"] = vlm_api
    key_args["renderer"] = renderer
    key_args["interaction_nb"] = interaction_nb

    return agent_map[agent_str](**key_args)
