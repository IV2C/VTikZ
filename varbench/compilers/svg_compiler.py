import PIL.Image
from . import Compiler
import cairosvg
import PIL
import io


class SvgCompiler(Compiler):
    def __init__(self):
        pass

    def compile(self, input: str, output: str):
        cairosvg.svg2png(url=input, write_to=output)

    def compile_from_string(self, input_string: str) -> PIL.Image:
        return PIL.Image.open(
            io.BytesIO(cairosvg.svg2png(bytestring=input_string.encode()))
        )
