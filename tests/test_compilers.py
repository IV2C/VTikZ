import unittest
import os
import PIL.Image
import timeout_decorator
import time
import numpy as np


class TestCompilers(unittest.TestCase):

    @timeout_decorator.timeout(600)
    def test_tex(self):
        from varbench.compilers import TexCompiler,CompilerException

        compiler = TexCompiler()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        output = os.path.join("tests/resources/tikz", "dog.jpeg")
        try:
            compiler.compile(tikzfile, output)
        except CompilerException as ce:
            print(ce)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(600)
    def test_svg(self):
        from varbench.compilers import SvgCompiler

        compiler = SvgCompiler()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        output = os.path.join("tests/resources/svg", "dog.jpeg")
        compiler.compile(svgFile, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(600)
    def test_tex_from_string(self):
        from varbench.compilers import TexCompiler

        compiler = TexCompiler()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        with open(tikzfile, "r") as f:
            tikzstring = f.read()

        result: PIL.Image = compiler.compile_from_string(tikzstring)
        self.assertTrue(np.any(np.array(result)))

    @timeout_decorator.timeout(600)
    def test_svg_from_string(self):
        from varbench.compilers import SvgCompiler

        compiler = SvgCompiler()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        with open(svgFile, "r") as f:
            svgstring = f.read()

        result: PIL.Image = compiler.compile_from_string(svgstring)
        self.assertTrue(np.any(np.array(result)))

    @timeout_decorator.timeout(600)
    def test_tex_from_string_exception(self):
        from varbench.compilers import TexCompiler, CompilerException

        compiler = TexCompiler()
        tikzfile = os.path.join("tests/resources/tikz", "malformed.tex")
        with open(tikzfile, "r") as f:
            tikzstring = f.read()

        self.assertRaises(CompilerException, compiler.compile_from_string, tikzstring)

    @timeout_decorator.timeout(600)
    def test_svg_from_string_exception(self):
        from varbench.compilers import SvgCompiler, CompilerException

        compiler = SvgCompiler()
        svgFile = os.path.join("tests/resources/svg", "malformed.svg")
        with open(svgFile, "r") as f:
            svgstring = f.read()

        self.assertRaises(CompilerException, compiler.compile_from_string, svgstring)

    def tearDown(self):
        if os.path.exists("tests/resources/tikz/dog.jpeg"):
            os.remove("tests/resources/tikz/dog.jpeg")
        if os.path.exists("tests/resources/svg/dog.jpeg"):
            os.remove("tests/resources/svg/dog.jpeg")


if __name__ == "__main__":
    unittest.main()
