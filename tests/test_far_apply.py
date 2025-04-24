import unittest
import re
import os
from vtikz.utils.parsing import apply_far_edit


class TestPatchApply(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.tikzfile = open(
            os.path.join("tests/resources/tikz", "simple_dog.tex")
        ).read()

        self.edit = self._get_first_code_block(
            open(os.path.join("tests/resources/tikz/edit_pupils.md")).read()
        )

    def _get_first_code_block(text):
        # Regular expression to find the first code block, ignoring the language specifier
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def test_far_patch(self):

        output = apply_far_edit(self.tikzfile, self.edit)
        self.assertEqual(open(os.path.join("tests/resources/tikz", "dog_pupils_edited.tex")).read(),output)
