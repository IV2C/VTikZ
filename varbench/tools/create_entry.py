import os
import argparse
from enum import Enum
import subprocess
from shutil import copytree,ignore_patterns
import logging

#enum for the type of subset
class SubsetType(Enum):
    TIKZ = "tikz"
    SVG = "svg"
    #other to add


parser = argparse.ArgumentParser()
parser.add_argument("subset",type=SubsetType)


args = parser.parse_args()

subset: SubsetType = args.subset


subset_path = os.path.join("dataset", subset.value)
entries_numbers = [entry.split("entry")[1] for entry in os.listdir(subset_path)]

if entries_numbers == []:
    entries_numbers = ["0"]

entries_numbers = [int(number) for number in entries_numbers]
#compute the next entry number(e.e. if only entry 1 and 3 exist, the next entry will be 2)
all_possible_number = list(range(1,max(entries_numbers)+2))
non_existent_entries = [number for number in all_possible_number if number not in entries_numbers]
new_entry_number = non_existent_entries[0]

new_entry_path = os.path.join(subset_path, f"entry{new_entry_number}")

os.mkdir(new_entry_path)

input_repo_path = os.path.join(new_entry_path, "input")

#note: pygit2 does not allow to clone a submodule with a different name, using subprocess instead

match subset:
    case SubsetType.TIKZ:
        subprocess.run(["git", "submodule", "add", "--name",f"tikzentry{new_entry_number}","--", "git@github.com:VarBench-SE/tikz.git", input_repo_path])
    case SubsetType.SVG:
        subprocess.run(["git", "submodule", "add", "--name",f"svgentry{new_entry_number}","--", "git@github.com:VarBench-SE/svg.git", input_repo_path])

copytree(input_repo_path, os.path.join(new_entry_path, "reference"),ignore=ignore_patterns("*.git"))
open(os.path.join(new_entry_path,"instruction.txt"), "w").close()