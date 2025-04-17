import hashlib
from typing import Iterable
import unittest
import uuid
from varbench.utils.caching import CachedRequest, instantiate_cache
from openai.types.chat import ChatCompletionMessageParam


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


cache = instantiate_cache(True, ".cache", "test_cache")


class TestCache(unittest.TestCase):
    class CachedClass:
        def __init__(self) -> None:
            self.temperature = 0.55
            self.model_name = "test_model_name"
            self.seed = 55
            self.n = 55
            super().__init__()

        @CachedRequest(key_function=key_function, cache=cache, is_enabled=True)
        def request(self, messages: Iterable[ChatCompletionMessageParam]):
            return uuid.uuid4()

    def test_cache_hit(self):
        cached_class = TestCache.CachedClass()
        first_response = cached_class.request(
            [{"role": "user", "content": "65qs4df1qs5df"}]
        )
        second_response = cached_class.request(
            [{"role": "user", "content": "eaazfasd"}]
        )
        cached_class.request([{"role": "user", "content": "qsdfdvds"}])
        cached_class.request([{"role": "user", "content": "cavwxcdf"}])

        self.assertEqual(
            cached_class.request([{"role": "user", "content": "65qs4df1qs5df"}]),
            first_response,
        )
        self.assertEqual(
            cached_class.request([{"role": "user", "content": "eaazfasd"}]),
            second_response,
        )

    def test_cache_miss(self):
        cached_class = TestCache.CachedClass()
        first_response = cached_class.request(
            [{"role": "user", "content": "65qs4df1qs5df"}]
        )
        second_response = cached_class.request(
            [{"role": "user", "content": "eaazfasd"}]
        )
        cached_class.request([{"role": "user", "content": "qsdfdvds"}])
        cached_class.request([{"role": "user", "content": "cavwxcdf"}])

        cached_class.seed = 56

        self.assertNotEqual(
            cached_class.request([{"role": "user", "content": "65qs4df1qs5df"}]),
            first_response,
        )
        self.assertNotEqual(
            cached_class.request([{"role": "user", "content": "eaazfasd"}]),
            second_response,
        )
