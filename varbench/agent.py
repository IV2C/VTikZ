from abc import ABC, abstractmethod
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

from varbench.api.chat_api import ChatApi


class Agent(ABC):
    def __init__(self,api:ChatApi,**kwargs) -> None:
        self.api = api
        pass

    @abstractmethod
    def compute(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs
    ) -> Iterable[str]:
        pass
    
    @abstractmethod
    def batchCompute(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        """Computes a list of responses from a list of messages, with the number of responses depending on the passk value set.

        Args:
            messages (Iterable[Iterable[ChatCompletionMessageParam]]): The list of messages
            ids (Iterable[str]): The list of associated Ids (usefull in case of openai batch api that put results unorderly)

        Returns:
            Iterable[Iterable[str]]: the list of list of messages
        """
        pass




class SimpleLLMAgent(Agent):        
    """Simple LLM agent that uses only the "reading" capabilities of the models tested

    Args:
        Agent (_type_): _description_
    """

    def compute(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs
    ) -> Iterable[str]:
        return self.api.chat_request(messages)
    
    def batchCompute(
        self,
        messages: Iterable[Iterable[ChatCompletionMessageParam]],
        ids: Iterable[str],
        **kwargs,
    ) -> Iterable[Iterable[str]]:
        return self.api.batch_chat_request(messages,ids)


