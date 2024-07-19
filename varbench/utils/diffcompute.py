"""given a folder of sets, compute the diff between the reference and input

usage:

python diffcompute.py --folder <folder>

where <folder> is the folder containing the sets, each set contains files named input and reference
"""


from diff_match_patch import diff_match_patch
import os
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()


diffObject = diff_match_patch()

for folder in os.listdir(args.folder):
    for set in os.listdir(os.path.join(args.folder, folder)):
        patch = diffObject.diff_main(
            open(os.path.join(args.folder, folder, set, "input")).read(),
            open(os.path.join(args.folder, folder, set, "reference")).read(),
        )
        real_patch = diffObject.patch_toText(patch)
        with open(os.path.join(args.folder, folder, set, "patch"), "w") as f:
            f.write(real_patch)
