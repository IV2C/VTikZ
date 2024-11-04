import PIL.Image
from . import Compiler,CompilerException
import cairosvg
import PIL
import io
from xml.etree.ElementTree import ParseError

class SvgCompiler(Compiler):
    def __init__(self):
        pass

    def compile(self, input: str, output: str):
        cairosvg.svg2png(url=input, write_to=output)

    def compile_from_string(self, input_string: str) -> PIL.Image:
        try:
            return PIL.Image.open(
            io.BytesIO(cairosvg.svg2png(bytestring=input_string.encode()))
        )
        except ParseError:
            raise CompilerException()
