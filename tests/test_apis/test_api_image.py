import os
import unittest
from loguru import logger
from pydantic import BaseModel, Field
from varbench.api.chat_api import ChatApi, GroqApi, OpenAIApi, VLLMApi
from varbench.utils.model_launch import launch_model


@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestApiCompletionImageGroq(unittest.TestCase):
    def test_request_n1_image(self):
        import base64
        from io import BytesIO
        from PIL import Image

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        api: ChatApi = GroqApi(0, 1, "llama-3.2-90b-vision-preview")

        response = api.chat_request(messages=messages)
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue("dog" in response[0])

    def test_structured_request_n1_image(self):
        logger.warning("no groq model supports image and structured output")
        pass


@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestApiCompletionImageVLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        launch_model("OpenGVLab/Mono-InternVL-2B")
        return super().setUpClass()

    def test_request_n1_image(self):
        import base64
        from io import BytesIO
        from PIL import Image

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        api: ChatApi = VLLMApi(0, 1, "OpenGVLab/Mono-InternVL-2B")

        response = api.chat_request(messages=messages)
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue("dog" in response[0])

    def test_structured_request_n1_image(self):

        import base64
        from io import BytesIO
        from PIL import Image

        api: ChatApi = VLLMApi(0, 1, "OpenGVLab/Mono-InternVL-2B")

        class Feature(BaseModel):
            feature: str
            description: str

        class Details(BaseModel):
            features: list[Feature]

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Detail the image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            }
        ]

        response = api.structured_request(messages=messages, response_format=Details)
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue(isinstance(response[0], Details))

@unittest.skipIf(os.environ.get("CI"), "Api tests skipped for CI")
class TestApiCompletionImageOpenAI(unittest.TestCase):
    def test_request_n1_image(self):
        import base64
        from io import BytesIO
        from PIL import Image

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        api: ChatApi = OpenAIApi(0, 1, "gpt-4o-mini")

        response = api.chat_request(messages=messages)
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue("dog" in response[0])

    def test_structured_request_n1_image(self):

        import base64
        from io import BytesIO
        from PIL import Image

        api: ChatApi = OpenAIApi(0, 1, "gpt-4o-mini")

        class Feature(BaseModel):
            name: str
            description: str

        class Details(BaseModel):
            features: list[Feature] = Field(description="a list of all the features in the image, with details about each feature")

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Detail the features of the image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            }
        ]

        response = api.structured_request(messages=messages, response_format=Details)
        logger.info(response)
        self.assertTrue(len(response) == 1)
        self.assertTrue(response[0] != None)
        self.assertTrue(isinstance(response[0], Details))
        
        
    def test_request_batch_n1_image(self):
        import base64
        from io import BytesIO
        from PIL import Image

        dog_image = Image.open("tests/resources/images/dog.jpeg")

        buffered = BytesIO()
        dog_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    },
                ],
            },
        ]

        api: ChatApi = OpenAIApi(1, 1, "gpt-4o-mini")

        response = api.batch_chat_request(messages=[messages,messages],ids=["first","second"])
        logger.info(response)
        self.assertTrue(len(response) == 2)
        self.assertTrue(response[0] != None)
        self.assertTrue("dog" in response[0] and "dog" in response[1])