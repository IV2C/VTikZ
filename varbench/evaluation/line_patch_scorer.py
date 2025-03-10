from typing import List
import re
from loguru import logger


def compute_line_score(
    prediction_diffs: List[List[str]], reference_diffs: List[List[str]]
) -> list[list[float]]:
    def _extract_modified_lines(diff_text: str) -> list[int]:
        # Regular expression to match the diff header for line numbers
        pattern_modified = r"@@ -(\d+(?:,\d+)?) \+\d+(?:,\d+)? @@"
        pattern_added = r"@@ -\d+(?:,\d+)? \+(\d+(?:,\d+)?) @@"
        matches_modified = re.findall(pattern_modified, diff_text)
        matches_added = re.findall(pattern_added, diff_text)
        # Convert matches to integers (line numbers), considering possible ranges
        modified_lines = []
        added_lines = []
        for match in matches_modified:
            # Split by comma and take the first number in the range
            splitted = match.split(",")
            number_of_lines = int(splitted[1] if len(splitted) > 1 else 1)
            line_modified = int(splitted[0])
            for i in range(number_of_lines):
                modified_lines.append(line_modified + i)
        for match in matches_added:
            # Split by comma and take the first number in the range
            splitted = match.split(",")
            number_of_lines = int(splitted[1] if len(splitted) > 1 else 1)
            line_modified = int(splitted[0])
            for i in range(number_of_lines):
                added_lines.append(line_modified + i)
        return set(modified_lines), set(added_lines).difference(
            set(modified_lines)
        )  # only added lines

    def _overlap(
        prediction: tuple[set[int], set[int]], reference: tuple[set[int], set[int]]
    ) -> float:
        # getting the biggest score from the comparison of the diff with the references diffs
        return (
            len(prediction[0] & reference[0]) + len(prediction[1] & reference[1])
        ) / (len(reference[0]) + len(reference[1]))

    # computing the line score(i.e. the lines' numbers)
    ## for each reference and prediction diffs get a list of the lines modified
    references_line_modified = [
        [_extract_modified_lines(diff) for diff in row_diffs]
        for row_diffs in reference_diffs
    ]
    predictions_line_modified = [
        [_extract_modified_lines(diff) for diff in row_diffs]
        for row_diffs in prediction_diffs
    ]

    # getting all the scores of each diff with the references
    all_pass_scores = [
        [
            [100 * _overlap(pred, refs) for pred in predictions_row_diffs]
            for refs in reference_row_diffs
        ]
        for predictions_row_diffs, reference_row_diffs in zip(
            predictions_line_modified, references_line_modified
        )
    ]

    # getting the max score that the predictions achieved
    return [
        [pass_score if pass_score else [0.0] for pass_score in pass_scores]
        for pass_scores in all_pass_scores
    ]
