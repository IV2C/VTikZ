import unittest
import os
import PIL.Image
import timeout_decorator
import time
import numpy as np


class TestRenderers(unittest.TestCase):

    @timeout_decorator.timeout(600)
    def test_tex(self):
        from varbench.renderers import TexRenderer, RendererException

        compiler = TexRenderer()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        output = os.path.join("tests/resources/tikz", "dog.jpeg")
        try:
            compiler.from_to_file(tikzfile, output)
        except RendererException as ce:
            print(ce)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(600)
    def test_svg(self):
        from varbench.renderers import SvgRenderer

        compiler = SvgRenderer()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        output = os.path.join("tests/resources/svg", "dog.jpeg")
        compiler.from_to_file(svgFile, output)
        self.assertTrue(os.path.exists(output), msg="Output file does not exist")

    @timeout_decorator.timeout(600)
    def test_tex_from_string(self):
        from varbench.renderers import TexRenderer

        compiler = TexRenderer()
        tikzfile = os.path.join("tests/resources/tikz", "dog.tex")
        with open(tikzfile, "r") as f:
            tikzstring = f.read()

        result: PIL.Image.Image = compiler.from_string_to_image(tikzstring)
        self.assertTrue(np.any(np.array(result)))

    @timeout_decorator.timeout(600)
    def test_svg_from_string(self):
        from varbench.renderers import SvgRenderer

        compiler = SvgRenderer()
        svgFile = os.path.join("tests/resources/svg", "dog.svg")
        with open(svgFile, "r") as f:
            svgstring = f.read()

        result: PIL.Image.Image = compiler.from_string_to_image(svgstring)
        self.assertTrue(np.any(np.array(result)))

    @timeout_decorator.timeout(600)
    def test_tex_from_string_exception(self):
        from varbench.renderers import TexRenderer, RendererException

        compiler = TexRenderer()
        tikzfile = os.path.join("tests/resources/tikz", "malformed.tex")
        with open(tikzfile, "r") as f:
            tikzstring = f.read()

        with self.assertRaises(RendererException) as context:
            compiler.from_string_to_image(tikzstring)

        self.assertEqual(
            context.exception.extract_error(),
            """! Extra }, or forgotten \\endgroup.
\\tikz@subpicture@handle@ ...interruptpath \\egroup
\\egroup \\egroup \\fi \\pgfke...
l.60 \\pic {squid}
;""",
        ) 

    @timeout_decorator.timeout(600)
    def test_svg_from_string_exception(self):
        from varbench.renderers import SvgRenderer, RendererException

        compiler = SvgRenderer()
        svgFile = os.path.join("tests/resources/svg", "malformed.svg")
        with open(svgFile, "r") as f:
            svgstring = f.read()
            
        with self.assertRaises(RendererException) as context:
            compiler.from_string_to_image(svgstring)


        self.assertEqual(
            context.exception.extract_error(),
            """mismatched tag: line 61, column 2""",
        ) 
        self.assertRaises(RendererException, compiler.from_string_to_image, svgstring)


    @timeout_decorator.timeout(600)
    def test_p5js_from_string(self):
        from varbench.renderers import P5JSRenderer

        compiler = P5JSRenderer()
        p5js_file = os.path.join("tests/resources/p5js", "kirby.js")
        with open(p5js_file, "r") as f:
            p5js_String = f.read()

        result: PIL.Image.Image = compiler.from_string_to_image(p5js_String)

        self.assertTrue(np.any(np.array(result)))
        
    @timeout_decorator.timeout(600)
    def test_P5js_from_string_exception(self):
        from varbench.renderers import P5JSRenderer, RendererException

        compiler = P5JSRenderer()
        p5js_file = os.path.join("tests/resources/p5js", "malformed.js")
        with open(p5js_file, "r") as f:
            p5js_string = f.read()
            
        with self.assertRaises(RendererException) as context:
            compiler.from_string_to_image(p5js_string)


        self.assertTrue(
            """Uncaught SyntaxError: Invalid or unexpected token""" in context.exception.extract_error()
            
        ) 
        self.assertRaises(RendererException, compiler.from_string_to_image, p5js_string)



    def tearDown(self):
        if os.path.exists("tests/resources/tikz/dog.jpeg"):
            os.remove("tests/resources/tikz/dog.jpeg")
        if os.path.exists("tests/resources/svg/dog.jpeg"):
            os.remove("tests/resources/svg/dog.jpeg")


if __name__ == "__main__":
    unittest.main()
