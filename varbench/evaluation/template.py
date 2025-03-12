import ast
import re
from varbench.dataset_workflow.utils import create_default

FLOAT_REG = r"(-?\d*\.?\d+)"
INTEGER_REG = r"(-?\d+)"
NON_RESERVED_LATEX_REG = r"([^#$%&_{}~^,\\\[\]\(\)]+)"
DEF_REG = (
    r"\§def\(" + NON_RESERVED_LATEX_REG + r"\)"
)  # Captures single parameter inside §def()
EQUAL_REG = (
    r"\§equal\(" + FLOAT_REG + r"\)"
)  # Captures single parameter inside §equal()
RANGE_REG = (
    r"\§range\(" + FLOAT_REG + r"," + FLOAT_REG + r"," + FLOAT_REG + r"\)"
)  # Captures three numerical parameters (floats or integers) inside §range()
RANGEI_REG = (
    r"\§rangei\("
    + FLOAT_REG
    + r","
    + FLOAT_REG
    + r"\)"  # Captures two integer parameters inside §rangei()
)

CHOICE_REG = (
    r"\§choice\(\[([^]]+)\],([^)]+)\)"  # Captures list and selected value in §choice()
)
PATTERNS = {
    "range": RANGE_REG,
    "rangei": RANGEI_REG,
    "choice": CHOICE_REG,
    "def": DEF_REG,
}


math_pattern = r"([+-]?\d*\.?\d+(?:[+\-\/*^]\d*\.?\d+)*)"


def evaluate_match(match):
    expr = match.group(1)
    try:
        result = round(eval(expr), 2)
        return str(int(result)) if result.is_integer() else str(result)
    except:
        return match.group(0)


def handle_rangei(
    prediction: int, start: int, end: int, args: tuple, all_matches: str
) -> tuple[str, str, list[tuple]]:
    return handle_range(
        prediction,
        start,
        end,
        (
            str(float(args[0]) - float(args[1])),
            str(float(args[0]) + float(args[1])),
            str(args[0]),
        ),
        all_matches,
    )

def handle_choice(
    prediction: int, start: int, end: int, args: tuple, all_matches: str
) -> tuple[str, str, list[tuple]]:
    replace_id = args[1]
    choices = [str(val) for val in ast.literal_eval(args[0])]
    prediction_def_start = prediction[start:]
    match = re.search(
        NON_RESERVED_LATEX_REG, prediction_def_start
    )  # matching the first non-latex expresssion in the prediction code
    if not match:
        return None, None, False
    match_value = match.group(1)
    if match_value in choices:
        prediction = (
            prediction[:start] + replace_id + prediction_def_start[match.end() :]
        )
        all_matches = update_matches(all_matches, end - start - len(replace_id), start)
        return prediction, all_matches, True
    else:
        return None, None, False



def handle_range(
    prediction: int, start: int, end: int, args: tuple, all_matches: str
) -> tuple[str, str, list[tuple]]:
    replace_id = str(args[2])
    prediction_def_start = prediction[start:]
    match = re.search(
        FLOAT_REG, prediction_def_start
    )  # matching the first non-latex expresssion in the prediction code
    if not match:
        return None, None, False
    match_value = float(match.group(1))
    if float(args[0]) <= match_value and float(args[1]) >= match_value:
        prediction = (
            prediction[:start] + replace_id + prediction_def_start[match.end() :]
        )
        all_matches = update_matches(all_matches, end - start - len(replace_id), start)
        return prediction, all_matches, True
    else:
        return None, None, False


def handle_def(
    prediction: str,
    start: int,
    end: int,
    args: tuple,
    all_matches: list[tuple],
) -> tuple[str, str, list[tuple]]:
    """handles ref command

    Args:
        template (str): original template code file
        prediction (str): initial prediction code file
        start (int): start of the template match
        end (int): end of the template match
        args (tuple): arguments from the match group
        all_matches (list[tuple]): all matches in the current template

    Returns:
        tuple[str,str,list[tuple]]: _description_
    """
    replace_id = args[0]
    prediction_def_start = prediction[start:]
    match = re.search(
        NON_RESERVED_LATEX_REG, prediction_def_start
    )  # matching the first non-latex expresssion in the prediction code
    if not match:
        return None, None, False
    var_name_prediction = match.group(1)
    all_matches = update_matches(all_matches, end - start - len(replace_id), start)
    prediction = prediction[:start] + replace_id + prediction_def_start[match.end() :].replace(var_name_prediction,replace_id)

    return prediction, all_matches, True


def update_matches(
    all_matches: list[tuple], len_to_ajust: int, found_index: int
) -> list[tuple]:
    return [
        (
            (start, end, key, args)
            if start < found_index
            else (start - len_to_ajust, end - len_to_ajust, key, args)
        )
        for (start, end, key, args) in all_matches
    ]


handle_map = {
    "range": handle_range,
    "rangei": handle_rangei,
    "choice": handle_choice,
    "def": handle_def,
}


def template_valid(template_code: str, prediction: str) -> bool:
    """Evaluates if prediction is valide with regard to the template code

    Args:
        template_code (str): prediction code
        prediction (str): template code

    Returns:
        bool: prediction valid or not.
    """
    prediction = re.sub(
        math_pattern, evaluate_match, prediction
    )  # evaluates mathematical expressions
    template_code = re.sub(
        math_pattern, evaluate_match, template_code
    )  # evaluates mathematical expressions
    matches = []
    for key, pattern in PATTERNS.items():

        for match in re.finditer(pattern, template_code, flags=re.DOTALL):
            start = match.start()
            groups = match.groups()
            end = match.end()
            if key in  "def":
                matches.append((start, end, key, [groups[0]]))  # Single captured value
            elif key == "range":
                matches.append(
                    (start, end, key, (groups[0], groups[1], groups[2]))
                )  # Start, End, default
            elif key == "rangei":
                matches.append(
                    (start, end, key, (groups[0], groups[1]))
                )  # value, interval
            elif key == "choice":
                matches.append(
                    (start, end, key, (groups[0], groups[1]))
                )  # option, default

    # handling other matches ordered
    ordered_matches = sorted(matches, key=lambda x: x[0])
    for i in range(len(ordered_matches)):
        start, end, key, args = ordered_matches[i]
        prediction, ordered_matches, ok = handle_map[key](
            prediction, start, end, args, ordered_matches
        )
        if not ok:
            return False
        i += 1  # Move to the next item

    if prediction == create_default(template_code):
        return True
    else:
        return False
