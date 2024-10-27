import json


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
        response = data["response"]["body"]["choices"][0]["message"]["content"]

        # Store the response associated with the custom_id
        if custom_id:
            custom_id_to_response[custom_id] = response
    return custom_id_to_response
