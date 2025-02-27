"""given a folder of sets, compute the diff between the input and solutions


the folder contains the subsets, each subsets contains the entries
"""

import os
import argparse

import difflib
import re

def patch_compute(folder: str)  -> tuple[str, str, str, str]:
    """
    Computes the differences between the 'input' and 'solutions' for each subset in the given folder.

    Parameters:
        folder (str): Path to the folder containing the 'input' and 'solutions' files.

    Returns:
        tuple:
            - str: The computed differences (unified diff format, without context).
            - str: The solution code (without comments).
            - str: The input code (without comments).
            - str: The original input file content (with comments).

    This function:
    - Identifies the 'input' and 'solutions' files in the given folder.
    - Removes comments from both files.
    - Computes a unified diff between the 'input' and 'solutions' files.

    Example usage:
        diff, solution, input_code, raw_input = patch_compute('/path/to/folder')
    """

    solution_folder = os.path.join(folder, "solutions")
    solution_path = os.listdir(solution_folder)[0]
    input_path = os.path.join(
        folder, [filename for filename in os.listdir(folder) if "input" in filename][0]
    )  # getting the input.? file


    with open(input_path) as input_file:
        commented_input = input_file.read()
        input = [re.split(r"(?<!\\)%", line)[0] for line in commented_input.splitlines() if not line.strip().startswith("%")]#removing comments
        input_code = "\n".join(input)
        with open(os.path.join(solution_folder, solution_path)) as solution_file:

            solution = "\n".join([re.split(r"(?<!\\)%", line)[0] for line in solution_file.read().splitlines() if not line.strip().startswith("%")])#removing comments
            
            solution_dif = solution.splitlines()
            current_diff = "".join(
                list(difflib.unified_diff(input, solution_dif, n=0))[2:]
            )  # getting the current diff without context

    return current_diff, solution,input_code,commented_input
