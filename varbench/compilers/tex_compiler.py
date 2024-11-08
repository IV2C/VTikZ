import PIL.Image
from . import Compiler
import os
import subprocess
from pdf2image import convert_from_path
from loguru import logger
from .compiler import CompilerException


class TexCompiler(Compiler):

    def __init__(self):
        super().__init__()
        pass

    def compile(self, input: str, output: str):
        output_cmd = subprocess.run(
            ["pdflatex", "-halt-on-error", "-output-directory", self.cache_path, input],
            capture_output=True,
        )
        if output_cmd.returncode != 0:
            raise CompilerException(output_cmd.stderr)

        output_file_name = os.path.join(
            self.cache_path, os.path.basename(input).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        image = convert_from_path(pdf_path=output_file_name)[0]
        image.save(output)

    def compile_from_string(self, input_string: str) -> PIL.Image:
        tmp_file_path = os.path.join(self.cache_path, "tmp.tex")
        file = open(tmp_file_path, "w")
        file.write(input_string)
        file.flush()
        file.close()
        output = subprocess.run(
            [
                "pdflatex",
                "-halt-on-error",
                "-output-directory",
                self.cache_path,
                tmp_file_path,
            ],
            capture_output=True,
        )
        if output.returncode != 0:
            raise CompilerException()
        
        output_file_name = os.path.join(
            self.cache_path, os.path.basename(tmp_file_path).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        return convert_from_path(pdf_path=output_file_name)[0]
