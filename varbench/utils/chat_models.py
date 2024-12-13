import json
import uuid

from pydantic import BaseModel


class ChatCompletionRequest:
    def __init__(
        self,
        messages: list,
        custom_id,
        response_format: BaseModel = None,
        model="gpt-3.5-turbo-0125",
        n=1,
        temperature=0,
        max_tokens=1000,
        seed =0
    ):
        self.custom_id = custom_id
        self.method = "POST"
        self.url = "/v1/chat/completions"
        self.body = {
            "model": model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed" :seed
        }
        if response_format:
            self.body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": type(response_format).__name__,
                    "schema": response_format.model_json_schema(),
                },
            }

    def to_json(self) -> str:
        return json.dumps(self.__dict__)
