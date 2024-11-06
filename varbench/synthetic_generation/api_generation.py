from openai import OpenAI
from groq import Groq
import instructor
from pydantic import BaseModel, Field
import os
from typing import List
import json


class VarbenchResponse(BaseModel):
    id: str = Field(
        ...,
        description='A representative identifier for the modification (for example "cat_grayed")',
    )
    instruction: str = Field(
        ..., description="An example of instruction that can be applied to the code"
    )
    result_description: str = Field(
        ...,
        description="A description of the result of the instruction applied to the code",
    )


class VarbenchResponses(BaseModel):
    modifications: List[VarbenchResponse] = Field(
        ...,
        description="list of modifications",
    )


def openai_generation_format(
    messages: list, model: str, temperature: int
) -> List[VarbenchResponse]:
    client = OpenAI()

    responses = client.beta.chat.completions.parse(
        temperature=temperature,
        model=model,
        messages=messages,
        response_format=VarbenchResponses,
    )
    return responses.modifications


def groq_generation_format(
    messages: list, model: str, temperature: int
) -> List[VarbenchResponse]:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    responses = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        response_model=VarbenchResponses,
    )
    with open("instructions.json", "w") as outjson:
        outjson.write(responses.model_dump_json(indent=2))

    return responses.modifications


def openai_generation(
    messages: list, model: str, temperature: int, n: int
) -> List[str]:
    client = OpenAI()

    completion = client.chat.completions.create(
        temperature=temperature, model=model, messages=messages, n=n
    )
    return [message.content for message in completion.choices]


def groq_generation(messages: list, model: str, temperature: int, n: int) -> List[str]:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    responses = [
        client.chat.completions.create(
            model=model, temperature=temperature, messages=messages
        )
        for _ in range(n)
    ]

    responses = [completion.choices[-1].message.content for completion in responses]

    with open("generation.json", "w") as jsonreponse:
        jsonreponse.write(json.dumps(responses))

    return responses
