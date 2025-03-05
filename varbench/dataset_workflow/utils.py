import re
import difflib
def unify_code(code:str):
        input = [re.split(r"(?<!\\)%", line)[0] for line in code.splitlines() if not line.strip().startswith("%")]#removing comments
        input = [re.sub(r"\s+", " ", line).strip() for line in input if line.strip()]  # Removing extra spaces and empty lines
        unified_code = "\n".join(input)
        return unified_code


def patch_compute(input: str, solution: str) -> str:

    solution_split = solution.splitlines()
    input_split = input.splitlines()
    current_diff = "".join(
        list(difflib.unified_diff(input_split, solution_split, n=0))[2:]
    )  # getting the current diff without context

    return current_diff

"""
templated_code => code with :
    §select(id){} 
    + all the parameterized
parameterized => code with:
    §optional(id){|optional code|} : code that serves no purpose but can still be there
    §range(lower,high,default): (math expr are evaluated)
    §rangei(value,interval) : defaults to value (math expr are evaluated)
    §choice([a,b,c],default)
    §def(var): defines var as a variable, can be reused elsewhere
    §exist(l1,l2){|ex|}: ex can exist between the lines provided in the file, whereas the other have to be at a specific place(must only be a single line)
    §equal(var): ensure any math expr is equal to the one specified(migh not be usefull) 
"""

def create_default(parameterized_code: str) -> str:
    """Creates a default code from a parameterized code
    note: preferably, the input should not be preprocessed(comments removed, etc), but the output should

    Args:
        parameterized_code (str): the input parameterized code

    Returns:
        str: parameterized code with default values
    """
    
    #simple default replacements
    parameterized_code = re.sub(r"§equal\((.*?)\)", r"\1", parameterized_code)
    parameterized_code = re.sub(r"§exist\([^,]+,[^)]+\)\{\|(.+?)\|\}", r"\1", parameterized_code, flags=re.DOTALL)
    parameterized_code = re.sub(r"§def\((.*?)\)", r"\1", parameterized_code)
    parameterized_code = re.sub(r"§range\([^,]+,[^,]+,([^)]+)\)", r"\1", parameterized_code)
    parameterized_code = re.sub(r"§rangei\(([^,]+),[^)]+\)", r"\1", parameterized_code)
    parameterized_code = re.sub(r"§choice\(\[[^]]+\],([^)]+)\)", r"\1", parameterized_code)
    
    default_code = re.sub(r'§optional\{\|.*?\|\}', '', parameterized_code, flags=re.DOTALL)
    
    return default_code

with open("/home/creux/Documents/AI/VariabilityBenchmark/dataset/tikz/bee_eyes/solutions/solution1.tex") as pt:
    print(create_default(pt.read()))