"""given a folder of sets, compute the diff between the input and solutions


the folder contains the subsets, each subsets contains the entries
"""

import os
import argparse

import difflib


def patch_compute(folder: str) -> list[str]:
    """
    Compute the differences between the 'input' and 'solutions' for each subset in the given folder.

    Parameters:
        folder (str): The path to the folder containing the input and solution(the entry in the dataset).

    Returns:
        None

    This function iterates over each subset and set in the given folder. For each set, it computes the differences between the 'input' and 'solutions'.

    Example usage:
        diffcompute('/path/to/folder')
    """

    solution_folder = os.path.join(folder, "solutions")
    solution_path = os.listdir(solution_folder)[0]
    input_path = os.path.join(
        folder, [filename for filename in os.listdir(folder) if "input" in filename][0]
    )  # getting the input.? file


    with open(input_path) as input_file:
        input = input_file.read().splitlines()

        with open(os.path.join(solution_folder, solution_path)) as solution_file:

            solution = solution_file.read()
            
            solution_dif = solution.splitlines()
            current_diff = "".join(
                list(difflib.unified_diff(input, solution_dif, n=0))[2:]
            )  # getting the current diff without context

    return current_diff, solution
