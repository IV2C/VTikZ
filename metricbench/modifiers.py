import re
import random
from metricbench.settings import TIKZ_COLORS
from PIL import Image
from varbench.renderers import Renderer
from varbench.renderers.renderer import RendererException


class TikzModifier:
    @staticmethod
    def _apply_random_match(code, pattern, replacement_func):
        """
        Helper to apply a modification to a random regex match.
        - `pattern`: Regex pattern to find matches.
        - `replacement_func`: Function to generate the replacement string for a match.
        """
        matches = list(re.finditer(pattern, code))
        if matches:
            chosen_match = random.choice(matches)
            start, end = chosen_match.start(), chosen_match.end()
            replacement = replacement_func(chosen_match)
            return code[:start] + replacement + code[end:]
        return code

    @staticmethod
    def change_color(code:str) -> str:
        # Replace a random color with a new one
        color_pattern = r"(" + "|".join(map(re.escape, TIKZ_COLORS)) + r")"

        return TikzModifier._apply_random_match(code, color_pattern, lambda _: "Teal500")

    @staticmethod
    def change_size(code:str) -> str:
        # Change the scale factor
        scale_pattern = r"(x=\d+cm/480,y=\d+cm/480)"
        return TikzModifier._apply_random_match(code, scale_pattern, lambda _: "x=5cm/480,y=5cm/480")

    @staticmethod
    def modify_radius(code:str) -> str:
        # Modify the radius
        radius_pattern = r"radius=(\d+)"
        return TikzModifier._apply_random_match(
            code, 
            radius_pattern, 
            lambda match: f"radius={int(match.group(1)) + random.randint(1, 10)}"
        )

    @staticmethod
    def add_shape(code:str) -> str:
        # Define random shapes, sizes, and colors
        SHAPES = ["circle", "rectangle", "ellipse"]
        SIZES = random.randint(5,30)
        COLORS = TIKZ_COLORS
        
        # Generate random shape, size, and color
        shape = random.choice(SHAPES)
        size = random.choice(SIZES)
        color = random.choice([color for color in COLORS if color in code])#getting only similar colors to the existing code
        x = random.randint(-150, 128)  # Random x-coordinate
        y = random.randint(-150, 128)  # Random y-coordinate
        
        
        # Construct TikZ code for the random shape
        if shape == "circle":
            shape_code = f"\\fill [{color}] ({x},{y}) circle [radius={size}pt];\n"
        elif shape == "rectangle":
            width, height = size, size // 2  # Rectangles are twice as wide as high
            shape_code = f"\\fill [{color}] ({x},{y}) rectangle +({width}pt,{height}pt);\n"
        elif shape == "ellipse":
            width, height = size, size // 2  # Ellipses are similar to rectangles
            shape_code = f"\\fill [{color}] ({x},{y}) ellipse [x radius={width}pt, y radius={height}pt];\n"
        else:
            return code  # Fallback in case of invalid shape

        # Insert the shape code randomly in the TikZ picture
        insert_points = [m.start() for m in re.finditer(r"\\fill|\\draw|\\pic", code)]
        if insert_points:
            insert_at = random.choice(insert_points)
            return code[:insert_at] + shape_code + code[insert_at:]
        else:
            insert_position = code.find("\\end{tikzpicture}")
            return code[:insert_position] + shape_code + code[insert_position:]
    @staticmethod
    def random_modification(code:str,renderer:Renderer) -> tuple[str,Image.Image]:
        methods = [
            TikzModifier.change_color,
            TikzModifier.change_size,
            TikzModifier.modify_radius,
            TikzModifier.add_shape
        ]
        chosen_method = random.choice(methods)
        
        new_code = chosen_method(code)
        while(True):
            try:
                image = renderer.from_string_to_image()
            
            except RendererException as e:
                pass        
        
        
