import unittest
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from pydantic import BaseModel
from vtikz.api import ChatApi
from loguru import logger
from vtikz.api.chat_api import OpenAIApi, ChatApi
import os

@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestApiCompletionOpenai(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
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
        api:ChatApi = OpenAIApi(0, 1, "gpt-4o-2024-08-06")

        ids = ["helpful_request", "unhelpful_request"]

        response = api.batch_chat_request(messages=self.chat_messages, ids=ids)
        logger.info(response)
        self.assertTrue(len(response) == 2)

    
    def test_api_batch_n2(self):
        api:ChatApi = OpenAIApi(0.8, 2, "gpt-4o-2024-08-06")
        ids = ["helpful_request", "unhelpful_request"]

        response = api.batch_chat_request(messages=self.chat_messages, ids=ids)
        logger.info(response)

        self.assertTrue(len(response) == len(response[0]) == len(response[1]) == 2)

    
    def test_structured_batch(self):
        api:ChatApi = OpenAIApi(0, 1, "gpt-4o-2024-08-06")

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
        api:ChatApi = OpenAIApi(0.8, 2, "gpt-4o-2024-08-06")
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
        api:ChatApi = OpenAIApi(0, 1, "gpt-4o-2024-08-06")

        response = api.chat_request(messages=self.chat_messages[0])
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)

    
    def test_request_n2(self):
        api:ChatApi = OpenAIApi(0.8, 2, "gpt-4o-2024-08-06")

        response = api.chat_request(messages=self.chat_messages[0])
        logger.info(response)
        self.assertTrue(len(response) == 2)
        self.assertTrue(response[0] != None and response[1] != None)

    
    def test_structured_request_n1(self):
        api:ChatApi = OpenAIApi(0, 1, "gpt-4o-2024-08-06")
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
        api:ChatApi = OpenAIApi(0.8, 2, "gpt-4o-2024-08-06")
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
