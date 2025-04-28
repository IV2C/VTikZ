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
import tqdm
import hashlib

from vtikz.utils.caching import CachedRequest, instantiate_cache
from vtikz.utils.chat_models import ChatCompletionRequest
from vtikz.utils.parsing import get_config, parse_openai_jsonl
import os
import pickle
import atexit

cache_enabled = get_config("MAIN").get("cache_enabled")

cache = instantiate_cache(
    cache_enabled, get_config("MAIN").get("cache_location", ".cache"), "chat_cache"
)

def key_function(func, *args, **kwargs):
    model_name = args[0].model_name
    temperature = args[0].temperature
    seed = args[0].seed
    n = args[0].n
    messages = str(args[1])
    func_name = func.__name__

    input_hash = hashlib.sha1(
        str((seed, model_name, temperature, messages, func_name, n)).encode("utf8")
    ).hexdigest()
    return input_hash


class ChatApi(ABC):
    def __init__(self, temperature: float, n: int, model_name: str) -> None:
        self.temperature = temperature
        self.n = n
        self.model_name = model_name
        self.seed = get_config("API").get("seed", 0)
        super().__init__()

    def from_url(
        temperature: float,
        n: int,
        model_name: str,
        api_url: str,
        api_key: str,
        *args,
        **kwargs,
    ) -> Self:
        if "groq" in api_url:
            logger.info("groq api setup")
            return GroqApi(temperature, n, model_name, api_url, api_key)
        elif "openai" in api_url:
            logger.info("openai api setup")
            return OpenAIApi(temperature, n, model_name, api_url, api_key)
        elif "localhost" in api_url or "runpod" in api_url:
            logger.info("vllm api setup")
            return VLLMApi(temperature, n, model_name, api_url, api_key)
        else:
            raise AttributeError(
                api_url, "Unsupported api, supported ones are vllm, groq, and openai"
            )  # TODO add "simple" openai chat api for compatible ones

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


from tqdm import tqdm


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
        self.client = OpenAI(base_url=api_url, api_key=api_key, timeout=2400.0)
        super().__init__(temperature, n, model_name)

    @CachedRequest(cache, key_function, cache_enabled)
    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        return [
            self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                n=1,
                seed=self.seed,
            )
            .choices[-1]
            .message.content
            for _ in range(self.n)
        ]

    @CachedRequest(cache, key_function, cache_enabled)
    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ) -> Iterable[BaseModel]:
        return [
            self.structured_client.chat.completions.create(
                model=self.model_name,
                response_model=response_format,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
            )
            for _ in range(self.n)
        ]

    def batch_chat_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return [
            self.chat_request(message)
            for message in tqdm(messages, desc="Request batch with groq api")
        ]

    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel = None,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        return [
            self.structured_request(message, response_format)
            for message in tqdm(messages, desc="Structured request batch with groq api")
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
        self.client = OpenAI(base_url=api_url, api_key=api_key, timeout=2400.0)
        super().__init__(temperature, n, model_name)

    @CachedRequest(cache, key_function, cache_enabled)
    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            n=self.n,
            seed=self.seed,
            max_completion_tokens=4096,
        )
        return [choice.message.content for choice in completion.choices]

    @CachedRequest(cache, key_function, cache_enabled)
    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            messages=messages,
            response_format=response_format,
            n=self.n,
            temperature=self.temperature,
            model=self.model_name,
            seed=self.seed,
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
                    seed=self.seed,
                    max_tokens=8192,
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
                    logger.error("OpenAI batch expired: " + str(batch_status))
                    return
                case "failed":
                    logger.error("Openai batched failed: " + str(batch_status))
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

    @CachedRequest(cache, key_function, cache_enabled)
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
            seed=self.seed,
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
        return [
            self.chat_request(message)
            for message in tqdm(messages, desc="Request batch with custom api")
        ]

    def batch_structured_request(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        response_format: BaseModel,
        **kwargs,
    ) -> Iterable[Iterable[BaseModel]]:
        return [
            self.structured_request(message, response_format)
            for message in tqdm(
                messages, desc="Structured request batch with custom api"
            )
        ]

    @CachedRequest(cache, key_function, cache_enabled)
    def chat_request(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> Iterable[str]:
        return [
            self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                n=1,
                seed=self.seed,
            )
            .choices[-1]
            .message.content
            for _ in range(self.n)
        ]

    @CachedRequest(cache, key_function, cache_enabled)
    def structured_request(
        self, messages: Iterable[ChatCompletionMessageParam], response_format: BaseModel
    ) -> Iterable[BaseModel]:
        return [
            self.structured_client.chat.completions.create(
                model=self.model_name,
                response_model=response_format,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
            )
            for _ in range(self.n)
        ]
