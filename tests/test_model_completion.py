import unittest
import os
import timeout_decorator
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class TestModelCompletion(unittest.TestCase):

    @unittest.skipIf(True, "paid test")
    def test_openai_batch(self):
        """This test uses the openai API, which uses money, hence the skipped test"""
        from varbench.model import API_model

        model: API_model = API_model("gpt-3.5-turbo-0125", 0)

        messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            [
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]

        ids = ["helpful_request", "unhelpful_request"]

        response = model.batchRequest(messages=messages, ids=ids)

        self.assertTrue(len(response) == 2)

    @unittest.skipIf(True, "paid test")
    def test_openai_batch_n2(self):
        """This test uses the openai API, which uses money, hence the skipped test"""
        from varbench.model import API_model

        model: API_model = API_model("gpt-3.5-turbo-0125", 1, 2)

        messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            [
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]

        ids = ["helpful_request", "unhelpful_request"]

        response = model.batchRequest(messages=messages, ids=ids)

        self.assertTrue(len(response) == len(response[0]) == len(response[1]) == 2)

    def test_vllm_batch(self):
        """This test uses the openai API, which uses money, hence the skipped test"""
        from varbench.model import VLLM_model

        model: VLLM_model = VLLM_model("meta-llama/Llama-3.2-1B-Instruct", 0)

        messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            [
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]

        ids = ["helpful_request", "unhelpful_request"]

        response = model.batchRequest(messages=messages, ids=ids)

        print(response)

        self.assertTrue(len(response) == 2)

    def test_vllm_batch_n2(self):
        """This test uses the openai API, which uses money, hence the skipped test"""
        from varbench.model import VLLM_model

        model: VLLM_model = VLLM_model("meta-llama/Llama-3.2-1B-Instruct", 1, n=2)

        messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            [
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        ]

        ids = ["helpful_request", "unhelpful_request"]

        response = model.batchRequest(messages=messages, ids=ids)

        print(response)

        self.assertTrue(len(response) == 2)


if __name__ == "__main__":
    unittest.main()
