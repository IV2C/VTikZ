from . import Compiler
from pdflatex import PDFLaTeX
import os
from pdf2image import convert_from_bytes



class TexCompiler(Compiler):

    def __init__(self):
        pass

    def compile(self, input: str, output: str):
        pdfl = PDFLaTeX.from_texfile(input)
        pdf,_,_ = pdfl.create_pdf()
        image = convert_from_bytes(pdf)[0]
        image.save(output)
        
    def compile_from_string(self, input_string: str, output: str):
        
        pdfl = PDFLaTeX.from_binarystring(input_string.encode(), 'dog.pdf')
        pdf,_,_ = pdfl.create_pdf()
        image = convert_from_bytes(pdf)[0]
        image.save(output)
        
