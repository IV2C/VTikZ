from . import Compiler
import os
import subprocess
from pdf2image import convert_from_path
from loguru import logger

class TexCompiler(Compiler):

    def __init__(self):
        super().__init__()
        pass

    def compile(self, input: str, output: str):
        logger.debug("Running pdfLatex:"+ " ".join(["pdflatex", "-output-directory", self.cache_path, input]))
        subprocess.run(
            ["pdflatex", "-output-directory", self.cache_path, input],
           
        )
        output_file_name = os.path.join(
            self.cache_path, os.path.basename(input).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        image = convert_from_path(pdf_path=output_file_name)[0]
        image.save(output)

    def compile_from_string(self, input_string: str, output: str):
        tmp_file_path = os.path.join(self.cache_path, "tmp.tex")
        file = open(tmp_file_path, "w")
        file.write(input_string)
        file.flush()
        file.close()
        self.compile(tmp_file_path, output)
