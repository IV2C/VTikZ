import unittest
import os
import timeout_decorator
import time


class TestCompilers(unittest.TestCase):

    @timeout_decorator.timeout(10)
    def test_tex(self):
        from varbench.compilers import TexCompiler

        compiler = TexCompiler()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        output = os.path.join("tests/resources/tikz", "dog.png")
        compiler.compile(tikzfile, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(10)
    def test_svg(self):
        from varbench.compilers import SvgCompiler

        compiler = SvgCompiler()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        output = os.path.join("tests/resources/svg", "dog.png")
        compiler.compile(svgFile, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    
    @timeout_decorator.timeout(10)
    def test_tex_from_string(self):
        from varbench.compilers import TexCompiler

        compiler = TexCompiler()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        with open(tikzfile, "r") as f:
            tikzstring = f.read()
        
        output = os.path.join("tests/resources/tikz", "dog.png")
        os.listdir("tests/resources/tikz")
        compiler.compile_from_string(tikzstring, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(10)
    def test_svg_from_string(self):
        from varbench.compilers import SvgCompiler

        compiler = SvgCompiler()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        with open(svgFile, "r") as f:
            svgstring = f.read()
        
        output = os.path.join("tests/resources/svg", "dog.png")
        compiler.compile_from_string(svgstring, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")
    

    def tearDown(self):
        if os.path.exists("tests/resources/tikz/dog.png"):
            os.remove("tests/resources/tikz/dog.png")
        if os.path.exists("tests/resources/svg/dog.png"):
            os.remove("tests/resources/svg/dog.png")
   


if __name__ == "__main__":
    unittest.main()
