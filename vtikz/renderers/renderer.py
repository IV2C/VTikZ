from abc import ABC, abstractmethod
import os

import PIL.Image

from vtikz.utils.parsing import get_config


class Renderer(ABC):
    def __init__(self):
        self.cache_path = os.path.join(os.environ.get("HOME"), ".cache/vtikz")
        self.rendering_timeout = get_config("RENDERER").get("timeout")
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

    @abstractmethod
    def from_to_file(self, input, output):
        pass

    @abstractmethod
    def from_string_to_image(self, input_string) -> PIL.Image.Image:
        pass

class RendererException(Exception):
    def __init__(self, message:str, *args: object) -> None:
        self.message = message
        super().__init__(*args)
    def __str__(self) -> str:
        return f"[RendererException:{self.message}]"
    def extract_error(self) -> str:
        """extracts the meaningful error from the error message 
        """
        pass
