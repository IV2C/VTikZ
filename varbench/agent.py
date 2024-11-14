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
    ) -> Iterable[str] | Iterable[BaseModel]:
        logger.info(f"Requesting to {self.client.base_url}")

        if response_format:
            #### With reponse format ####
            if ApiType.OpenAI:
                chat_function = self.client.beta.chat.completions.parse
            else:
                chat_function = self.client.chat.completions.create
            parameters = {
                "messages": messages,
                "stop": ["\n```\n"],
                "model": self.model_name,
                "temperature": self.temperature,
            }
            if self.apitype == ApiType.Groq:
                parameters["response_model"] = response_format
                parameters["n"] = 1
            elif self.apitype == ApiType.VLLM or self.apitype == ApiType.OpenAI:
                parameters["response_format"] = response_format
                parameters["n"] = self.n

            match self.apitype:
                case ApiType.Groq:
                    return [chat_function(**parameters) for _ in range(self.n)]
                case _:
                    ...
        else:
            ####Without Structured Output####
            chat_function = self.client.chat.completions.create
            if self.apitype == ApiType.Groq:
                return [
                    chat_function(
                        messages=messages,
                        stop=["\n```\n"],
                        model=self.model_name,
                        temperature=self.temperature,
                        n=1,
                        response_format=response_format,
                    )
                    .choices[-1]
                    .message.content
                    for _ in range(self.n)
                ]
            completion = chat_function(
                messages=messages,
                stop=["\n```\n"],
                model=self.model_name,
                temperature=self.temperature,
                n=self.n,
            )

        if response_format:
            if ApiType.OpenAI:
                return [choice.message.parsed for choice in completion.choices]
            elif ApiType.Groq:
                return []
        else:
            return [choice.message.content for choice in completion.choices]

    def batchRequest(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[Iterable[str]] | Iterable[Iterable[BaseModel]]:

        # using single requests if no batch
        if self.no_batch or self.apitype == ApiType.Groq:
            return [
                self.request(messages_n, response_format=response_format)
                for messages_n in messages
            ]

        # Using the openai batch api otherwise

        ## creating message completion objects
        batch_str = "\n".join(
            [
                ChatCompletionRequest(
                    messages=message,
                    custom_id=id,
                    n=self.n,
                    response_format=response_format,
                ).to_json()
                for message, id in zip(messages, ids)
            ]
        )
        ## creating "false" file to provide to the api
        batch_file = BytesIO(batch_str.encode())
        batch_input_file = self.client.files.create(file=batch_file, purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "VarBench eval job"},
        )
        batch_id = batch.id
        ## loop for response handling
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            logger.debug("Current_status:" + batch_status.status)
            file_id = batch_status.output_file_id
            match (batch_status.status):
                case "completed":
                    file_response = self.client.files.content(file_id)
                    logger.info(file_response.response)
                    logger.info(file_response)
                    simplified_response = parse_openai_jsonl(file_response.text)
                    return [
                        simplified_response[id] for id in ids
                    ]  # returns the ids in order
                case "expired":
                    logger.error("OpenAI batch expired: " + batch_status)
                    return
                case "failed":
                    logger.error("Openai batched failed: " + batch_status)
                case _:
                    sleep(300)


from enum import Enum


