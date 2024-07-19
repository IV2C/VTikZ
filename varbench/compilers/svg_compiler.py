from . import Compiler
import cairosvg


class SvgCompiler(Compiler):
    def __init__(self):
        pass
    def compile(self, input: str, output: str):
        cairosvg.svg2png(url=input, write_to=output)
        pass
    
    def compile_from_string(self, input_string: str, output: str):
        cairosvg.svg2png(bytestring=input_string.encode(), write_to=output)
        pass