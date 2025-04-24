from PIL import Image
import subprocess

from vtikz.utils.parsing import get_config


from . import Renderer, RendererException


import uuid
import PIL.Image
import os
import subprocess
import shutil


class P5JSRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.cache_path = os.path.join(self.cache_path, "p5")
        os.makedirs(self.cache_path, exist_ok=True)
        self.browser = {**get_config("RENDERER")}["p5js_browser_path"]

        shutil.copytree(
            "vtikz/renderers/resources",
            self.cache_path + "/resources",
            dirs_exist_ok=True,
        )

    def from_to_file(self, input: str, output: str):
        raise NotImplementedError()
        # TODO

    def from_string_to_image(self, input_string: str) -> PIL.Image.Image:
        current_cache = os.path.join(self.cache_path, str(uuid.uuid4()))
        current_html_path = os.path.join(current_cache, "index.html")
        os.mkdir(current_cache)
        shutil.copyfile(
            os.path.join(self.cache_path, "resources", "index.html"),
            current_html_path,
        )
        sketch_path = os.path.join(current_cache, "sketch.js")
        with open(sketch_path, "w") as sketch:
            sketch.write(input_string)
        url = "file:///" + current_html_path
        image_cache_path = os.path.join(current_cache, "output.png")
        output = subprocess.run(
            [
                f"{self.browser}/chrome",
                "--headless",
                "--screenshot=" + image_cache_path,
                "--windows-size=1920,1080",
                url,
                "--default-background-color=00000000",
                "--hide-scrollbars",
                "--enable-logging=stderr",
                '--js-flags="--turboprop"',
                "--v=1",
            ],
            capture_output=True,
        )
        errorlines = [
            line
            for line in str(output.stderr).split("\\n")
            if (sketch_path in line and "INFO:CONSOLE" in line)
        ]
        if len(errorlines) > 0:
            raise P5JSRendererException(str(errorlines))

        img = Image.open(image_cache_path)
        img = img.convert("RGBA")
        bbox = img.getbbox()
        cropped_img = img.crop(bbox)
        shutil.rmtree(current_cache)
        return cropped_img


class P5JSRendererException(RendererException):
    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)

    def __str__(self) -> str:
        return f"[P5JSRendererException:{self.message}]"

    def extract_error(self) -> str:
        return self.message
