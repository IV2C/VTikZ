from datasets import Dataset
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import subprocess
import os
from ..prompt_templates import *


def clone_subset_repository(subset_str):
    """clones the subset repository

    Args:
        subset_str (str): the subset repository to clone
    """
    subprocess.run(
        [
            "git",
            "clone",
            f"https://github.com/VarBench-SE/{subset_str}",
            os.path.join("eval", subset_str),
        ]
    )


def evaluate_tikz(subset: Dataset, generator, it_tuned: bool, generation_config):
    """Evaluates a model on the TikZ subset.

    Args:
        subset (Dataset): The subset to evaluate the model on.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model for evaluation.
        it_tuned (bool): Flag to toggle between instruction-tuned and non-tuned modes.
    """

    def tikz_dataset_explorer(subset: Dataset):
        clone_subset_repository("tikz")

        for entry in subset:
            instruction = entry["instruction"]
            commit = entry["commit_id"]

            # Switch to the appropriate git commit
            try:
                subprocess.run(
                    ["git", "checkout", commit],
                    cwd=os.path.join("eval", "tikz"),
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error checking out commit {commit}: {e}")
                continue

            # Get the first .tex file and read content
            try:
                tikz_file = next(
                    file
                    for file in os.listdir(os.path.join("eval", "tikz"))
                    if file.endswith(".tex")
                )
                with open(os.path.join("eval", "tikz", tikz_file), "r") as f:
                    content = f.read()
                    # Generate full prompt
                    full_prompt = (IT_PROMPT if it_tuned else PROMPT).format(
                        instruction=instruction, content=content
                    )
            except (StopIteration, FileNotFoundError) as e:
                print(f"Error processing files in commit {commit}: {e}")
                continue

            print(f"FULL PROMPT:\n{full_prompt}")
            yield full_prompt

    # Tokenize the prompt and generate output using the model
    for prompt in tikz_dataset_explorer(subset):
        generated_text = generator(prompt, **generation_config)
        print(generated_text)


def evaluate_svg(subset: Dataset, generator, it_tuned: bool):
    """evaluates a model on the svg subset

    Args:
        subset (Dataset): the subset to evaluate the model on
        pipeline (Pipeline): the model to evaluate
    """


evaluation_dispatcher = {"tikz": evaluate_tikz, "svg": evaluate_svg}
