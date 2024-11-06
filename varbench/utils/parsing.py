import json
import re

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
        responses = [response["message"]["content"] for response in data["response"]["body"]["choices"]]

        # Store the response associated with the custom_id
        if custom_id:
            custom_id_to_response[custom_id] = responses
    return custom_id_to_response

def get_first_code_block(text):
    # Regular expression to find the first code block, ignoring the language specifier
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None