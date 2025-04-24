import PIL.Image
from . import Renderer,RendererException
import cairosvg
import PIL
import io
from xml.etree.ElementTree import ParseError
from loguru import logger
class SvgRenderer(Renderer):
    def __init__(self):
        pass

    def from_to_file(self, input: str, output: str):
        cairosvg.svg2png(url=input, write_to=output)

    def from_string_to_image(self, input_string: str) -> PIL.Image.Image:
        try:
            return PIL.Image.open(
            io.BytesIO(cairosvg.svg2png(bytestring=input_string.encode()))
        )
        except ParseError as pe:
            raise SvgRendererException(pe.msg)
        
class SvgRendererException(RendererException):
    def __init__(self, message:str, *args: object) -> str:
        logger.warning(message)
        super().__init__(str(message),*args)
    def __str__(self) -> str:
        return f"[SvgRendererException:{self.message}]"
    def extract_error(self):
        """extracts the meaningful error from the error message 
        """
        return self.message