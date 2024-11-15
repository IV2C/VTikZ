from io import BufferedReader, BytesIO
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, wait_exponential

from varbench.utils.parsing import get_config, parse_openai_jsonl
from enum import Enum


class ApiType(Enum):
    Groq = "Groq"
    VLLM = "VLLM"
    OpenAI = "OpenAI"


def get_api_type(api_string):
    if "groq" in api_string.lower():
        return ApiType.Groq
    elif "localhost" in api_string.lower():
        return ApiType.VLLM
    elif "openai" in api_string.lower():
        return ApiType.OpenAI
    else:
        raise Exception("unsupport API with url " + api_string)


class Agent:
    def __init__(self, model_name, temperature, n=1, **kwargs) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.n = n

    def request(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs
    ) -> Iterable[str]:
        pass

    def batchRequest(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        pass


from openai import OpenAI
import os
from varbench.utils.chat_models import ChatCompletionRequest
from time import sleep
from typing import Any
import instructor
from groq import Groq


class SimpleLLMAgent(Agent):
    def __init__(
        self,
        model_name,
        temperature,
        n=1,
        api_url=None,
        api_key=None,
        no_batch=False,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, n)
        self.no_batch = no_batch
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        # in the case of groq api, we use instructor for structured outputs
        self.apitype = get_api_type(api_url)
        match self.apitype:
            case ApiType.Groq:
                self.structured_client = instructor.from_groq(
                    Groq(api_key=api_key),
                    mode=instructor.Mode.JSON,
                )
                self.client = OpenAI(
                    base_url=api_url,
                    api_key=api_key,
                )
            case ApiType.VLLM | ApiType.OpenAI:
                self.structured_client = self.client = OpenAI(
                    base_url=api_url,
                    api_key=api_key,
                )

    # @retry(wait=wait_exponential(multiplier=1, min=4))
    def request(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[str] :
       pass
    def batchRequest(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        pass



