
import hashlib

from loguru import logger
from typing import Iterable
from PIL import Image
from vif_agent.agent import VifAgent
from varbench.agents.external.external_agent import ExternalAgent
from varbench.renderers import TexRenderer
from varbench.utils.caching import CachedRequest, instantiate_cache
from varbench.utils.parsing import get_config
from openai import OpenAI

cache_enabled = get_config("VIF").get("cache_enabled")
cache = instantiate_cache(
    cache_enabled,
    get_config("VIF").get("cache_location", ".cache"),
    "vif",
)


def key_function(func, *args, **kwargs):
    agent_params = str(args[0].agent)
    n = args[0].n
    seed = args[0].seed
    instruction = str(args[1])
    code = args[2]
    func_name = func.__name__

    input_hash = hashlib.sha1(
        str((agent_params, instruction, func_name, n, seed, code)).encode("utf8")
    ).hexdigest()
    return input_hash


class VIFAgent(ExternalAgent):
    def __init__(self, **kwargs):
        renderer = TexRenderer()  # hardcoded for now
        vif_args = {**get_config("VIF")}
        client = OpenAI(api_key=vif_args["api_key"], base_url=vif_args["api_url"])
        identification_client = OpenAI(
            api_key=vif_args["search_api_key"], base_url=vif_args["search_api_url"]
        )
        search_client = OpenAI(
            api_key=vif_args["identification_api_key"],
            base_url=vif_args["identification_api_url"],
        )
        self.seed = vif_args["seed"]
        self.agent = VifAgent(
            code_renderer=renderer.from_string_to_image,
            client=client,
            model=vif_args["model"],
            search_client=search_client,
            search_model=vif_args["search_model"],
            identification_client=identification_client,
            identification_model=vif_args["identification_model"],
            temperature=vif_args["temperature"],
        )
        self.n = kwargs["n"]
        super().__init__(**kwargs)

    @CachedRequest(cache, key_function, cache_enabled)
    def compute(
        self, instruction: str, code: str, image: Image.Image = None, **kwargs
    ) -> Iterable[str]:
        return [self.agent.apply_instruction(code, instruction) for _ in range(self.n)]

    def batchCompute(
        self,
        instructions: Iterable[str],
        codes: Iterable[str],
        ids: Iterable[str],
        image_input: Iterable[Image.Image] = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        responses = [
            self.compute(instruction, code)
            for instruction, code in zip(instructions, codes)
        ]
        return responses
