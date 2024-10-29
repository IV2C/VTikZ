import json
import uuid


class ChatCompletionRequest:
    def __init__(
        self, messages: list, custom_id, model="gpt-3.5-turbo-0125",n=1,temperature=0, max_tokens=1000
    ):
        self.custom_id = custom_id
        self.method = "POST"
        self.url = "/v1/chat/completions"
        self.body = {"model": model, "messages": messages,"n":n,"temperature":temperature,"max_tokens":max_tokens}

    def to_json(self) -> str:
        return json.dumps(self.__dict__)
