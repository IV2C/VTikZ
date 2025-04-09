import configparser
import json
import re
from loguru import logger


def parse_openai_jsonl(input_data: str) -> dict[str, str]:
    # Split the JSONL input into separate lines
    lines = input_data.strip().splitlines()

    # Initialize a dictionary to store custom_id and response mappings
    custom_id_to_response = {}

    # Process each line as a JSON object
    for line in lines:
        # Parse the JSON object from the line
        data = json.loads(line)

        # Extract the custom_id and response
        custom_id = data["custom_id"]
        responses = [
            response["message"]["content"]
            for response in data["response"]["body"]["choices"]
        ]

        # Store the response associated with the custom_id
        if custom_id:
            custom_id_to_response[custom_id] = responses
    return custom_id_to_response


def get_first_code_block(text):
    # Regular expression to handle spaces and optional language specifier
    match = re.search(r"```(?:[a-zA-Z]*\n)?([^`]*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def replace_first_code_block(text, replacement):
    pattern = r"^```(?:[a-zA-Z]*\n)?(.+?)^```$"

    try:
        match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)

        if match:
            full_match = match.group(0)
            lang_match = re.match(r"```([a-zA-Z]*)\n", full_match)
            lang = lang_match.group(1) if lang_match else ""

            replacement_block = f"```{lang}\n{replacement}\n```"

            return text.replace(full_match, replacement_block, 1)

        return text

    except re.error as e:
        print(f"Regex error: {e}")
        return text


def make_numerical(string_value: str) -> float | int | str | bool:
    try:
        result = int(string_value)
    except ValueError:
        try:
            result = float(string_value)
        except ValueError:
            if str(string_value) == "True":
                return True
            elif str(string_value) == "False":
                return False

            return string_value
    return result


def get_config(config_name: str):
    config = configparser.ConfigParser()
    config.read("config-varbench.cfg")
    config_k = {
        key: make_numerical(value) for key, value in config[config_name].items()
    }
    return config_k


def apply_far_edit(content: str, far_edit: str):
    # Split the instructions on '---' and '+++', so that we capture find and replace parts
    changes = far_edit.strip().split("---")[
        1:
    ]  # Skip the first empty part caused by the first '---'

    for change in changes:
        # Split the change into before (find) and after (replace) based on the '+++' marker
        parts = change.split("+++")

        if len(parts) == 2:
            before_change = parts[0].strip()
            after_change = parts[1].strip()

            # Apply the find and replace to the original text
            content = content.replace(before_change, after_change)

    return content


def _valid_model_name(model_name):
    return model_name.replace("/", "").replace("-", "").replace(":", "")


def get_config_name(args, split_used):

    if args.agent == "VIF":  # External agent
        vif_args = {**get_config("VIF")}
        full_config_name = (
            args.agent
            + "_"
            + split_used
            + "_"
            + _valid_model_name(vif_args["model"])
            + "_t_"
            + str(vif_args["temperature"])
            + "_search_"
            + _valid_model_name(vif_args["search_model"])
            + "_t_"
            + str(vif_args["search_model_temperature"])
            + "_identification_"
            + _valid_model_name(vif_args["identification_model"])
            + "_t_"
            + (str(vif_args["identification_model_temperature"]))
        )
    else:  # internal_agent
        full_config_name = (
            args.agent
            + "_"
            + split_used
            + "_"
            + _valid_model_name(args.model)
            + "_pk_"
            + str(args.passk)
            + "_t_"
            + str(args.temperature)
            + (
                (
                    "_"
                    + _valid_model_name(args.vlm)
                    + "_t_"
                    + str(args.vlm_temperature)
                )
                if args.vlm
                else ""
            )
        )

    return full_config_name
