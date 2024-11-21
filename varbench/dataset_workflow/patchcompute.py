"""given a folder of sets, compute the diff between the input and solutions


the folder contains the subsets, each subsets contains the entries
"""
import os
import argparse

import difflib

def patch_compute(folder:str)-> list[str]:
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
    
    solution_folder = os.path.join(folder,"solutions")
    
    input_path = os.path.join(folder,[filename for filename in os.listdir(folder) if "input" in filename][0])#getting the input.? file
    
    solutions = []
    
    with open(input_path) as input_file:
        input = input_file.read().splitlines()
        diffs:set = set()
        for solution in os.listdir(solution_folder):
            with open(os.path.join(solution_folder,solution)) as solution_file:
                
                solution = solution_file.read()
                solutions.append(solution)
                solution = solution.splitlines()
                current_diff = "".join(list(difflib.unified_diff(input, solution, n=0))[2:])#getting the current diff without context
                diffs.add(current_diff)
                
    return list(diffs),solutions
