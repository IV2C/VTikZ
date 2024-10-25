from io import BufferedReader, BytesIO
from typing import Iterable
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class LLM_Model:
    def __init__(self) -> None:
        pass

    def request(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        pass

    def batchRequest(
        self, messages: Iterable[Iterable[ChatCompletionMessageParam]], **kwargs
    ) -> Iterable[str]:
        pass


from vllm import LLM, SamplingParams


class VLLM_model(LLM_Model):
    def __init__(self, model_name, temperature, gpu_number=1, **kwargs) -> None:
        self.model_name = model_name
        self.llm: LLM = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=gpu_number,
            gpu_memory_utilization=0.9,
            **kwargs
        )
        self.temperature = temperature

    def request(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        return self.batchRequest(messages=[messages], **kwargs)[0]

    def batchRequest(
        self, messages: Iterable[Iterable[ChatCompletionMessageParam]], **kwargs
    ) -> Iterable[str]:
        self.samplingParams: SamplingParams = SamplingParams(
            temperature=self.temperature, stop="\n```\n", **kwargs
        )
        outputs = self.llm.chat(messages=messages, sampling_params=self.samplingParams)
        return [
            "".join([output.text for output in completion.outputs])
            for completion in outputs
        ]


from openai import OpenAI
import os
from utils.chat_models import ChatCompletionRequest
from time import sleep

class API_model(LLM_Model):
    def __init__(
        self, model_name, temperature, api_url="https://api.openai.com/v1", api_key=None
    ) -> None:
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.temperature = temperature

    def request(self, messages: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        completion = self.client.chat.completions.create(
            messages=messages, stop=["\n```\n"], **kwargs
        )
        return completion.choices[-1].message.content

    def batchRequest(
        self, messages: Iterable[Iterable[ChatCompletionMessageParam]], **kwargs
    ) -> Iterable[str]:
        batch_str = "\n".join(
            [ChatCompletionRequest(messages=message).to_json() for message in messages]
        )
        batch_file = BytesIO(batch_str)
        batch_input_file = self.client.files.create(file=batch_file, purpose="batch")
        batch_input_file_id = batch_input_file.id

        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "varbench eval job"},
        )
        batch_id = batch.id
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            match (batch_status.status):
                case "completed":
                    file_response = self.client.files.content("file-xyz123")
        pass


from enum import Enum


class ModelType(Enum):
    API = 0
    VLLM = 1
