import PIL.Image
from . import Renderer
import os
import subprocess
from pdf2image import convert_from_path
from loguru import logger
from .renderer import RendererException


class TexRenderer(Renderer):

    def __init__(self):
        super().__init__()
        pass

    def from_to_file(self, input: str, output: str):
        output_cmd = subprocess.run(
            ["pdflatex", "-halt-on-error", "-output-directory", self.cache_path, input],
            capture_output=True,
        )
        if output_cmd.returncode != 0:
            raise RendererException(output_cmd.stderr)

        output_file_name = os.path.join(
            self.cache_path, os.path.basename(input).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        image = convert_from_path(pdf_path=output_file_name)[0]
        image.save(output)

    def from_string_to_image(self, input_string: str) -> PIL.Image.Image:
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
            raise TexRendererException(
                output.stderr.decode() + "|" + output.stdout.decode()
            )

        output_file_name = os.path.join(
            self.cache_path, os.path.basename(tmp_file_path).replace("tex", "pdf")
        )

        logger.debug("converting to png")
        return convert_from_path(pdf_path=output_file_name)[0]


class TexRendererException(RendererException):
    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)

    def __str__(self) -> str:
        return f"[TexRendererException:{self.message}]"

    def extract_error(self)->str:
        error_lines = []
        start_saving = False
        exception_message = self.message.split("\n")
        for line in exception_message:
            if line.startswith("! "):  # start of error message
                start_saving = True

            if start_saving:
                if line.startswith(
                    "!  ==> Fatal error occurred"
                ):  # end of error message
                    start_saving = False
                    continue
                error_lines.append(line.strip())
        return "\n".join(error_lines)
