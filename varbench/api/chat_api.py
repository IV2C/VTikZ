from abc import ABC, abstractmethod
from io import BytesIO
from time import sleep
from typing import Iterable, Self
from groq import Groq
import instructor
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from openai import OpenAI
from loguru import logger

from varbench.utils.chat_models import ChatCompletionRequest
from varbench.utils.parsing import get_config, parse_openai_jsonl
import os


class ChatApi(ABC):
    def __init__(self, temperature: float, n: int, model_name: str) -> None:
        self.temperature = temperature
        self.n = n
        self.model_name = model_name
        self.stop = ["\n```\n"]
        super().__init__()

    def from_url(
        temperature: float, n: int, model_name: str, api_url: str, api_key: str,*args, **kwargs
    ) -> Self:
        if "groq" in api_url:
            logger.info("groq api setup")
            return GroqApi(temperature, n, model_name, api_url, api_key)
        elif "openai" in api_url:
            logger.info("openai api setup")
            return OpenAIApi(temperature, n, model_name, api_url, api_key)
        elif "localhost" in api_url:
            logger.info("vllm api setup")
            return VLLMApi(temperature, n, model_name, api_url, api_key)
        else:
            raise AttributeError(api_url,"Unsupported api, supported ones are vllm, groq, and openai")#TODO add "simple" openai chat api for compatible ones

    @abstractmethod
    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        pass

    @abstractmethod
    def structured_request(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        response_format: BaseModel,
    ) -> Iterable[BaseModel]:
        pass

    @abstractmethod
    def batch_chat_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        pass

    @abstractmethod
    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        pass


class GroqApi(ChatApi):
    def __init__(
        self,
        temperature: float,
        n: int,
        model_name: str,
        api_url: str = "https://api.groq.com/openai/v1",
        api_key: str = os.environ.get("GROQ_API_KEY"),
    ) -> None:
        self.structured_client = instructor.from_groq(
            Groq(api_key=api_key), mode=instructor.Mode.JSON
        )
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        super().__init__(temperature, n, model_name)

    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        return [
            self.client.chat.completions.create(
                messages=messages,
                stop=self.stop,
                model=self.model_name,
                temperature=self.temperature,
                n=1,
            )
            .choices[-1]
            .message.content
            for _ in range(self.n)
        ]

    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ) -> Iterable[BaseModel]:
        return [
            self.structured_client.chat.completions.create(
                model=self.model_name,
                response_model=response_format,
                messages=messages,
                temperature=self.temperature,
            )
            for _ in range(self.n)
        ]

    def batch_chat_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [self.chat_request(message) for message in messages]

    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        return [
            self.structured_request(message, response_format) for message in messages
        ]


class OpenAIApi(ChatApi):

    def __init__(
        self,
        temperature: float,
        n: int,
        model_name: str,
        api_url: str = "https://api.openai.com/v1",
        api_key: str = os.environ.get("OPENAI_API_KEY"),
    ) -> None:
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        super().__init__(temperature, n, model_name)

    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        completion = self.client.chat.completions.create(
            messages=messages,
            stop=self.stop,
            model=self.model_name,
            temperature=self.temperature,
            n=self.n,
        )
        return [choice.message.content for choice in completion.choices]

    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=response_format,
            n=self.n,
            temperature=self.temperature,
            model=self.model_name,
        )
        return [choice.message.parsed for choice in completion.choices]

    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        batch_str = "\n".join(
            [
                ChatCompletionRequest(
                    messages=message,
                    custom_id=id,
                    n=self.n,
                    response_format=response_format,
                    model=self.model_name,
                    temperature=self.temperature,
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
                    simplified_response = parse_openai_jsonl(file_response.text)

                    ordered_responses = [
                        simplified_response[id] for id in ids
                    ]  # returns the ids in order
                    if response_format == None:
                        return ordered_responses
                    return [
                        [
                            response_format.model_validate_json(res)
                            for res in cur_responses
                        ]
                        for cur_responses in ordered_responses
                    ]
                case "expired":
                    logger.error("OpenAI batch expired: " + batch_status)
                    return
                case "failed":
                    logger.error("Openai batched failed: " + batch_status)
                case _:
                    sleep(500)

    def batch_chat_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return self.batch_structured_request(messages, ids, None, **kwargs)


class VLLMApi(OpenAIApi):
    def __init__(
        self,
        temperature: float,
        n: int,
        model_name: str,
        api_url: str = "http://localhost:8056/v1",
        api_key: str = None,
    ) -> None:
        api_key = api_key or get_config("VLLM").get("api-key", "vllm_key_not_set")

        super().__init__(temperature, n, model_name, api_url, api_key)

    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ):
        completion = self.client.chat.completions.create(
            messages=messages,
            extra_body=dict(
                guided_json=response_format.model_json_schema(),
                guided_decoding_backend="outlines",
            ),
            n=self.n,
            temperature=self.temperature,
            model=self.model_name,
        )
        return [
            response_format.model_validate_json(choice.message.content)
            for choice in completion.choices
        ]

    def batch_chat_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [self.chat_request(message) for message in messages]

    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        return [
            self.structured_request(message, response_format) for message in messages
        ]
