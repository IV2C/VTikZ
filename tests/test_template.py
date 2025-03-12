import unittest
from varbench.evaluation.template import (
    CHOICE_REG,
    RANGE_REG,
    RANGEI_REG,
    handle_choice,
    handle_def,
    DEF_REG,
    handle_range,
    template_valid,
)
import re


class TestTemplate(unittest.TestCase):

    def test_def_replace(self):
        prediction = """
\\definecolor{green}{rgb}{0.9, 0.13, 0.13}
this
green
test green
"""
        template = """
\\definecolor{§def(blue)}{rgb}{0.9, 0.13, 0.13}
this
blue
test blue
"""
        expected = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
this
blue
test blue
"""
        all_matches = [(3, 5, None, None), (10, 12, None, None), (55, 66, None, None)]
        found_index = re.search(DEF_REG, template)
        new_prediction, new_matches, ok = handle_def(
            prediction,
            found_index.start(),
            found_index.end(),
            found_index.groups(),
            all_matches,
        )
        self.assertEqual(expected, new_prediction)

    def test_range_replace(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        expected = """
\\definecolor{blue}{rgb}{1, 0.13, 0.13}
"""
        all_matches = [(3, 5, None, None), (10, 12, None, None), (55, 66, None, None)]
        found_index = re.search(RANGE_REG, template)
        print(found_index.groups())
        new_prediction, new_matches, ok = handle_range(
            prediction,
            found_index.start(),
            found_index.end(),
            found_index.groups(),
            all_matches,
        )
        self.assertEqual(expected, new_prediction)

    def test_range_replace_outside(self):
        prediction = """
\\definecolor{blue}{rgb}{1.2, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        all_matches = [(3, 5, None, None), (10, 12, None, None), (55, 66, None, None)]
        found_index = re.search(RANGE_REG, template)
        print(found_index.groups())
        new_prediction, new_matches, ok = handle_range(
            prediction,
            found_index.start(),
            found_index.end(),
            found_index.groups(),
            all_matches,
        )
        self.assertFalse(ok)

    def test_choice_replace(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
"""
        expected = """
\\definecolor{blue}{rgb}{1.1, 0.13, 0.13}
"""
        all_matches = [(3, 5, None, None), (10, 12, None, None), (55, 66, None, None)]
        found_index = re.search(CHOICE_REG, template)
        print(found_index.groups())
        new_prediction, new_matches, ok = handle_choice(
            prediction,
            found_index.start(),
            found_index.end(),
            found_index.groups(),
            all_matches,
        )
        self.assertEqual(expected, new_prediction)

    def test_choice_replace_outside(self):
        prediction = """
\\definecolor{blue}{rgb}{1.0, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
"""
        all_matches = [(3, 5, None, None), (10, 12, None, None), (55, 66, None, None)]
        found_index = re.search(CHOICE_REG, template)
        print(found_index.groups())
        new_prediction, new_matches, ok = handle_choice(
            prediction,
            found_index.start(),
            found_index.end(),
            found_index.groups(),
            all_matches,
        )
        self.assertFalse(ok)

    def test_template_range(self):
        prediction = """
\\definecolor{blue}{rgb}{1.1, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_range_invalid(self):
        prediction = """
\\definecolor{blue}{rgb}{1.2, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_rangei(self):
        prediction = """
\\definecolor{blue}{rgb}{1, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§rangei(0.9,0.1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_rangei_invalid(self):
        prediction = """
\\definecolor{blue}{rgb}{1.1, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§rangei(0.9,0.1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_choice(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_choice_invalid(self):
        prediction = """
\\definecolor{blue}{rgb}{1.0, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_choice_str(self):
        prediction = """
\\definecolor{blue}{rgb}{valid}
"""
        template = """
\\definecolor{blue}{rgb}{§choice(["valid","vali","alid"],alid)}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_choice_invalid_str(self):
        prediction = """
\\definecolor{blue}{rgb}{invalid}
"""
        template = """
\\definecolor{blue}{rgb}{§choice(["valid","vali","alid"],alid)}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_choice_range(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
\\definecolor{blue}{rgb}{1.0, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_choice_range_invalid(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
\\definecolor{blue}{rgb}{1.2, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice([0.9,1.1],1.1), 0.13, 0.13}
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_choice_range_str(self):
        prediction = """
\\definecolor{blue}{rgb}{valid}
\\definecolor{blue}{rgb}{1.1, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§choice(["valid","vali","alid"],alid)}
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))


    def test_template_range_rangei(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
\\definecolor{blue}{rgb}{1.0, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
\\definecolor{blue}{rgb}{§rangei(0.9,0.1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_range_rangei_invalid(self):
        prediction = """
\\definecolor{blue}{rgb}{0.9, 0.13, 0.13}
\\definecolor{blue}{rgb}{1.2, 0.13, 0.13}
"""
        template = """
\\definecolor{blue}{rgb}{§range(0.9,1.1,1), 0.13, 0.13}
\\definecolor{blue}{rgb}{§rangei((0.9,0.1),1.1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_def_range(self):
        prediction = """
\\definecolor{red}{rgb}{0.9, 0.13, 0.13}
\\fill{red}{1.0, 0.13, 0.13}
"""
        template = """
\\definecolor{§def(blue)}{rgb}{0.9, 0.13, 0.13}
\\fill{blue}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertTrue(template_valid(template, prediction))

    def test_template_def_range_invalid(self):
        prediction = """
\\definecolor{red}{rgb}{0.9, 0.13, 0.13}
\\fill{red}{1.3, 0.13, 0.13}
"""
        template = """
\\definecolor{§def(blue)}{rgb}{0.9, 0.13, 0.13}
\\fill{blue}{§range(0.9,1.1,1), 0.13, 0.13}
"""
        self.assertFalse(template_valid(template, prediction))

    def test_template_complex(self):
        prediction = """
\definecolor{laserG}{rgb}{0.2, 0.6, 0.1}
\draw [fill = laserG, draw = laserG] (-1,8) circle [radius=0.1];
\draw [fill = laserG, draw = laserG] (-1,8) circle [radius=0.1];
\definecolor{laserR}{rgb}{0.95-0.05, 0+.1, 0.2}
\draw [fill = laserR, draw = laserR] (-1,8) circle [radius=0.1];
\draw [fill = laserG, draw = laserR] (-1,8) circle [radius=0.1];
\definecolor{laserB}{rgb}{0.25, 0.75, 0.5+0.3}
\draw [fill = laserG, draw = laserR] (-1,8) circle [radius=0.1];
\draw [fill = laserB, draw = laserB] (-1,8) circle [radius=0.1];
"""
        template = """
\definecolor{§def(laserGreen)}{rgb}{§range(0,0.3,0), §rangei(0.7,0.2), §choice([0,0.1,0.2,0.3],0.3)}
\draw [fill = laserGreen, draw = laserGreen] (-1,8) circle [radius=0.1];
\draw [fill = laserGreen, draw = laserGreen] (-1,8) circle [radius=0.1];
\definecolor{§def(laserRed)}{rgb}{§range(.9,1,0.9), §rangei(0.1,0.1), §choice([0,0.1,0.2,0.3],0.3)}
\draw [fill = laserRed, draw = laserRed] (-1,8) circle [radius=0.1];
\draw [fill = laserGreen, draw = laserRed] (-1,8) circle [radius=0.1];
\definecolor{§def(laserBlue)}{rgb}{§range(0,0.3,0), §rangei(0.7,0.2), §choice([0.7,0.8,0.9,1],0.9)}
\draw [fill = laserGreen, draw = laserRed] (-1,8) circle [radius=0.1];
\draw [fill = laserBlue, draw = laserBlue] (-1,8) circle [radius=0.1];
"""
        self.assertTrue(template_valid(template, prediction))


    def test_template_full_file(self):
        template = open("tests/resources/tikz/template/pupils_dog_template.tex").read()
        prediction = open("tests/resources/tikz/template/valid_to_template.tex").read()
        self.assertTrue(template_valid(template, prediction))

    def test_no_param_template(self):
        template = open("tests/resources/tikz/template/no_param_template.tex").read()
        self.assertTrue(template_valid(template, template))
        