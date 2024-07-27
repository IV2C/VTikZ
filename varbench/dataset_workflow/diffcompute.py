"""given a folder of sets, compute the diff between the reference and input

usage:

python diffcompute.py --folder <folder>

where <folder> is the folder containing the splits, each splits contains the sets
"""

import os
import argparse

import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()


for split in os.listdir(args.folder):
    for set in os.listdir(os.path.join(args.folder, split)):
        with open(os.path.join(args.folder, split, set, f"{set}.patch"),"w") as f:
            subprocess.run(
                [
                    "diff",
                    "-u",
                    os.path.join(args.folder, split, set, "input"),
                    os.path.join(
                        args.folder,
                        split,
                        set,
                        "reference"
                    ),
                    "--exclude",
                    ".git",
                ],stdout=f
            )
