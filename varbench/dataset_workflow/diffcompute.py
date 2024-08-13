"""given a folder of sets, compute the diff between the reference and input


the folder contains the subsets, each subsets contains the entries
"""

import os
import argparse

import subprocess


def diffcompute(folder):
    """
    Compute the differences between the 'input' and 'reference' directories for each set in the given folder.

    Parameters:
        folder (str): The path to the folder containing the subsets and sets.

    Returns:
        None

    This function iterates over each subset and set in the given folder. For each set, it computes the differences between the 'input' and 'reference' directories using the 'diff' command. The differences are written to a file named '{set}.patch' in the corresponding set directory. The '--exclude' option is used to exclude the '.git' directory from the comparison.

    Note: The 'subprocess.run' function is used to execute the 'diff' command. The 'stdout' parameter is set to the file object 'f' to write the output directly to the file.

    Example usage:
        diffcompute('/path/to/folder')
    """
    for subset in os.listdir(folder):
        for entry in os.listdir(os.path.join(folder, subset)):
            with open(os.path.join(folder, subset, entry, f"{entry}.patch"), "w") as f:
                subprocess.run(
                    [
                        "diff",
                        "-u",
                        "input",
                        "reference",
                        "--exclude",
                        ".git",
                    ],
                    cwd=os.path.join(folder, subset, entry),
                    stdout=f,
                )
