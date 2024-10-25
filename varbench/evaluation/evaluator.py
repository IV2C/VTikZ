import difflib
from datasets import Dataset
import subprocess
import os
from ..prompt_templates import *
from ..model import LLM_Model

def evaluate(subset: Dataset,model:LLM_Model):
    
    # Tokenize the prompt and generate output using the model
    subset.map



def _compute(self, inputs, predictions, diffs):
    """Returns the scores"""
    # TODO: Compute the different scores of the module

    varscore = sum(
        "".join(list(difflib.unified_diff(i, p, n=0))[2:]) in d
        for i, p, d in zip(inputs, predictions, diffs)
    ) / len(predictions)
    return {"varscore": varscore}


