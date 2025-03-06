import re

EXIST_REG = r"\§exist\((-?\d+),(-?\d+)\)\{\|(.+?)\|\}"  # Captures all parameters inside §exist()
EQUAL_REG = r"\§equal\((-?\d*\.?\d+)\)"  # Captures single parameter inside §equal()
DEF_REG = r"\§def\([^#$%&_{}~^,\\]\)"  # Captures single parameter inside §def()
RANGE_REG = r"\§range\((-?\d*\.?\d+),(-?\d*\.?\d+),(-?\d*\.?\d+)\)"  # Captures three numerical parameters (floats or integers) inside §range()
RANGEI_REG = (
    r"\§rangei\((-?\d+),(-?\d+)\)"  # Captures two integer parameters inside §rangei()
)
CHOICE_REG = (
    r"\§choice\(\[([^]]+)\],([^)]+)\)"  # Captures list and selected value in §choice()
)
OPTIONAL_REG = r"§optional\{\|(.*?)\|\}"  # Captures content in optional
PATTERNS = {
    "equal": EQUAL_REG,
    "def": DEF_REG,
    "range": RANGE_REG,
    "rangei": RANGEI_REG,
    "choice": CHOICE_REG,
    "optional": OPTIONAL_REG,
}
math_pattern = r"([+-]?\d*\.?\d+(?:[+-\/*^]\d*\.?\d+)*)"


def evaluate_match(match):
    expr = match.group(1)
    try:
        result = str(eval(expr))
        return result
    except:
        return match.group(0)


def template_valid(template_code: str, prediction: str):
    prediction = re.sub(
        math_pattern, evaluate_match, prediction
    )  # evaluates mathematical expressions

    # handling exists first, as they can be at different locations in the code

    exist_matches = re.findall(EXIST_REG, template_code, flags=re.DOTALL)
    template_code, prediction, valid = handle_exists(
        exist_matches, template_code, prediction
    )
    if not valid:
        return False
    matches = {}

    for key, pattern in PATTERNS.items():
        for match in re.finditer(pattern, template_code, flags=re.DOTALL):
            start = match.start()
            groups = match.groups()
            end = match.end()
            if key == "exist":
                matches.append(
                    (start, end, key, (groups[0], groups[1], groups[2]))
                )  # l1, l2, Content
            elif key in {"equal", "def", "optional"}:
                matches.append((start, end, key, (groups[0])))  # Single captured value
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
    ordered_matches = [
        match for match in sorted(matches, key=lambda x: x[0]) if match[2] != "exist"
    ]
    for start, end, key, args in ordered_matches:
        match key:
            case "range":
                handle_range()


def handle_exists(matches: list, template_code: str, prediction: str):
    splitted_prediction = prediction.splitlines()
    for start, end, _, (l1, l2, content) in matches:
        reduced_prediction = "\n".join(splitted_prediction[l1 - 1 : l2 - 1])
        splitted_prediction = find_and_replace_def_in_exist(content,prediction,reduced_prediction)#start by replacing all defs by the default value
        reduced_prediction = "\n".join(splitted_prediction[l1 - 1 : l2 - 1])
        regex = create_regex_from_exist(content)#no more def, no groups found
        match_list = list(re.finditer(regex, reduced_prediction))
        if len(match_list) == 0:
            return None, None, False
        else:
            match = match_list[0]
            template_code = template_code[:start] + template_code[end:]
            reduced_prediction = (
                reduced_prediction[: match.start()] + reduced_prediction[match.end() :]
            )
            splitted_prediction = (
                splitted_prediction[: l1 - 1]
                + reduced_prediction.splitlines()
                + splitted_prediction[l2 - 1 :]
            )
            removed_length = match.end() - match.start()
            matches = [
                (s - removed_length, e - removed_length, k, args)
                for s, e, k, args in matches
            ]

    return template_code, "\n".join(splitted_prediction), True


EQUAL_REG = r"\§equal\(-?\d*\.?\d+\)"
DEF_REG = r"\§def\([^#$%&_{}~^,\\])"
RANGE_REG = r"\§range\(-?\d*\.?\d+,-?\d*\.?\d+,(?\d*\.?\d+\)"
RANGEI_REG = r"\§rangei\(-?\d+,-?\d+\)"
CHOICE_REG = r"\§choice\(\[[^]]+\],[^)]+\)"


def find_and_replace_def_in_exist(content: str, prediction: str, reduced_prediction: str)->list[str]:
    """Finds definitions of variables in a exist command, and replace it in all the rest of the code

    Args:
        content (str): content in the exist
        prediction (str): entire prediction code
        reduced_prediction (str): prediction where the match can occur

    Returns:
        list[str]: _description_
    """
    found_def_identifiers = re.findall(DEF_REG, content)
    ex_regex = create_regex_from_exist(content)
    match = list(re.finditer(ex_regex, reduced_prediction))[0]
    if len(match.groups()) > 0:  # found a def
        for id, group in zip(found_def_identifiers, match.groups()):
            prediction = (
                prediction.replace(group, id)
            )
    return prediction.splitlines()


def create_regex_from_exist(content: str):
    reg = content
    reg = re.sub(EQUAL_REG, r"-?\d*\.?\d+", reg)
    reg = re.sub(DEF_REG, r"([^#$%&_{}~^,\\])", reg)
    reg = re.sub(RANGE_REG, r"-?\d*\.?\d+", reg)
    reg = re.sub(RANGEI_REG, r"-?\d*\.?\d+", reg)
    reg = re.sub(CHOICE_REG, r"[^#$%&_{}~^,\\]", reg)
    return reg


def handle_range(
    index: int, start: int, end: int, default: str, prediction: str, template_code: str
) -> str:
    pass

    # remove the optional if exists
    # create a regex to find the pattern of the code inside the exist within the line args
    # if exists remove the pattern from both the template and prediction and iterate until either no more exist or pattern not found => return false
    # do re.match for any command:
    # get the index of the match(Match.start())
    # verify that the expression in the prediction at that index matches the command => yes replace with default in both template_code and prediction / no return false
    # §def(value) => replace in the rest of the entire prediction code the value defined in the prediction by the value of the template
    # §range(low,hi,default) => find expression at index with regex,evaluate expression at index, then if >low <high replace with default else false
    # §rangei(value,interval) => find expression at index with regex,evaluate expression at index, then if >value-interval <value+interval replace with default else false
    # §choice([a,b,..],default)) => find expression at index with regex, not there false, there replace with default
    # §equal(exp) => evaluate expression at index , = exp is replace with exp
    # redo until no more command
    # once no more command strict equality between both codes
