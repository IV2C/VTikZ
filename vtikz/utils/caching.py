import atexit
import os
import pickle

from loguru import logger


def instantiate_cache(is_enabled: bool, cache_path: str, name: str):
    if is_enabled:

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        cache_location = os.path.join(cache_path, name)

        if os.path.exists(cache_location):
            cache_chatapi = open(cache_location, "rb")
            cache: dict[int, str] = pickle.load(cache_chatapi)
        else:
            cache: dict[int, str] = {}

        def save_cache(cache, cache_location):
            logger.info("program exited, saving cache")
            pickle.dump(cache, open(cache_location, "wb"))

        logger.info(f"{name} cache loaded")
        atexit.register(lambda: save_cache(cache, cache_location))
    return cache


def CachedRequest(cache, key_function, is_enabled):
    def innerCachedRequest(func):
        def checkForCached(*args, **kwargs):
            if not is_enabled:
                return func(*args, **kwargs)
            input_key = key_function(func, *args, **kwargs)

            return_value = cache.get(input_key)
            if not return_value:
                return_value = func(*args, **kwargs)
                cache[input_key] = return_value
                logger.debug("new cache")
                logger.debug(str(cache))
            else:
                logger.warning("Cache hit")
            return return_value

        return checkForCached

    return innerCachedRequest
