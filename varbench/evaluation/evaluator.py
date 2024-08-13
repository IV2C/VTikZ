from datasets import Dataset
from transformers import Pipeline
import subprocess
import os
def clone_subset_repository(subset_str):
    """clones the subset repository

    Args:
        subset_str (str): the subset repository to clone
    """
    subprocess.run(["git", "clone", f"https://github.com/VarBench-SE/{subset_str}", os.path.join("eval",subset_str)])
    

def evaluate_tikz(subset: Dataset, pipeline: Pipeline):
    """evaluates a model on the tikz subset

    Args:
        subset (Dataset): the subset to evaluate the model on
        pipeline (Pipeline): the model to evaluate
    """
    def tikz_dataset_explorer(subset: Dataset):
        clone_subset_repository("tikz")
        for entry in subset:
            prompt = entry["prompt"]
            commit = entry["commit_id"]
            subprocess.run(["git","checkout",commit],cwd=os.path.join("eval","tikz"))
            tikz_file = [file for file in os.listdir(os.path.join("eval","tikz")) if file.endswith(".tex")][0]
            content = open(os.path.join("eval","tikz",tikz_file)).read()
            full_prompt = "exercice: " + prompt + "\n\n" + content + "\n\n" + "answer:"
            print("FULL PROMPT:\n"+full_prompt)
            yield full_prompt
    for out in pipeline(tikz_dataset_explorer(subset)):
        print(out)
    
def evaluate_svg(subset: Dataset, pipeline: Pipeline):
    """evaluates a model on the svg subset
    
    Args:
        subset (Dataset): the subset to evaluate the model on
        pipeline (Pipeline): the model to evaluate
    """  


evaluation_dispatcher = {
    "tikz": evaluate_tikz,
    "svg": evaluate_svg
}
