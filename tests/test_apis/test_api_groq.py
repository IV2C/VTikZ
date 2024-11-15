import unittest
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import os

from pydantic import BaseModel
from varbench.api import ChatApi
from loguru import logger
from varbench.api.chat_api import GroqApi, GroqApi, VLLMApi, ChatApi

from varbench.utils.model_launch import launch_model

@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestApiCompletionGroq(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        # launch_model("meta-llama/Llama-3.2-1B-Instruct")
        self.chat_messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            [
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]
        super().__init__(methodName)

    
    def test_api_batch(self):
        api:ChatApi = GroqApi(0, 1, "llama-3.1-70b-versatile")

        ids = ["helpful_request", "unhelpful_request"]

        response = api.batch_chat_request(messages=self.chat_messages, ids=ids)
        logger.info(response)
        self.assertTrue(len(response) == 2)

    
    def test_api_batch_n2(self):
        api:ChatApi = GroqApi(0.8, 2, "llama-3.1-70b-versatile")
        ids = ["helpful_request", "unhelpful_request"]

        response = api.batch_chat_request(messages=self.chat_messages, ids=ids)
        logger.info(response)

        self.assertTrue(len(response) == len(response[0]) == len(response[1]) == 2)

    
    def test_structured_batch(self):
        api:ChatApi = GroqApi(0, 1, "llama-3.1-70b-versatile")

        class Operation(BaseModel):
            operation: str
            result: int

        ids = ["helpful_request", "unhelpful_request"]

        response: list[list[Operation]] = api.batch_structured_request(
            messages=self.chat_messages, ids=ids, response_format=Operation
        )
        logger.info(response)
        self.assertTrue(response[0][0].operation != None)
        self.assertTrue(response[0][0].result != None)
        self.assertTrue(len(response) == 2)

    
    def test_structured_batch_n2(self):
        api:ChatApi = GroqApi(0.8, 2, "llama-3.1-70b-versatile")
        class Operation(BaseModel):
            operation: str
            result: int

        ids = ["helpful_request", "unhelpful_request"]

        response: list[list[Operation]] = api.batch_structured_request(
            messages=self.chat_messages, ids=ids, response_format=Operation
        )
        logger.info(response)
        self.assertTrue(response[0][0].operation != None)
        self.assertTrue(response[0][0].result != None)
        self.assertTrue(len(response) == len(response[0]) == len(response[1]) == 2)

    
    def test_request_n1(self):
        api:ChatApi = GroqApi(0, 1, "llama-3.1-70b-versatile")

        response = api.chat_request(messages=self.chat_messages[0])
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)

    
    def test_request_n2(self):
        api:ChatApi = GroqApi(0.8, 2, "llama-3.1-70b-versatile")

        response = api.chat_request(messages=self.chat_messages[0])
        logger.info(response)
        self.assertTrue(len(response) == 2)
        self.assertTrue(response[0] != None and response[1] != None)

    
    def test_structured_request_n1(self):
        api:ChatApi = GroqApi(0, 1, "llama-3.1-70b-versatile")
        class Operation(BaseModel):
            operation: str
            result: int

        response = api.structured_request(
            messages=self.chat_messages[0], response_format=Operation
        )
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue(isinstance(response[0], Operation))

    
    def test_structured_request_n2(self):
        api:ChatApi = GroqApi(0.8, 2, "llama-3.1-70b-versatile")
        class Operation(BaseModel):
            operation: str
            result: int

        response = api.structured_request(
            messages=self.chat_messages[0], response_format=Operation
        )
        logger.info(response)
        self.assertTrue(len(response) == 2)
        self.assertTrue(response[0] != None and response[1] != None)
        self.assertTrue(isinstance(response[0], Operation))


if __name__ == "__main__":
    unittest.main()
