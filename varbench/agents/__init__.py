from .agent import Agent
from .code_Correct_Agent import CodeCorrectAgent
from .LMM_loop_agent import LMMLoopAgent
from .simple_LLM_agent import SimpleLLMAgent
from .LMM_agent import LMMAgent
from .VLLM_loop_agent import VLLMLoopAgent
from .FAR_LLM_Agent import FARAgent

def instantiate_agent(
    agent_str: str,
    api,
    vlm_api=None,
    renderer=None,
    interaction_nb=3,
) -> Agent:
    agent_map = {
        "simpleLLM": SimpleLLMAgent,
        "simpleLMM": LMMAgent,
        "loopVLMLLM": VLLMLoopAgent,
        "loopLMM": LMMLoopAgent,
        "FAR": FARAgent
    }

    key_args = {}
    key_args["api"] = api
    key_args["vlm_api"] = vlm_api
    key_args["renderer"] = renderer
    key_args["interaction_nb"] = interaction_nb

    return agent_map[agent_str](**key_args)
