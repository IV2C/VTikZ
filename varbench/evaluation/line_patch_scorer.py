from typing import List
import re


def compute_line_score(
    prediction_diffs: List[List[str]], reference_diffs: List[str]
) -> list[list[float]]:
    def _extract_modified_lines(diff_text: str) -> list[int]:
        # Regular expression to match the diff header for line numbers
        pattern = r"@@ -(\d+(?:,\d+)?) \+\d+(?:,\d+)? @@"
        matches = re.findall(pattern, diff_text)
        # Convert matches to integers (line numbers), considering possible ranges
        modified_lines = []
        for match in matches:
            # Split by comma and take the first number in the range
            modified_lines.append(int(match.split(',')[0]))
        return modified_lines

    def _overlap(prediction: set[int], reference: set[int]) -> float:
        # getting the biggest score from the comparison of the diff with the references diffs
        return len(prediction & reference) / len(reference) 
        


    # computing the line score(i.e. the lines' numbers)
    ## for each reference and prediction diffs get a list of the lines modified
    reference_line_modified = [
        set(_extract_modified_lines(row_diff))
        for row_diff in reference_diffs
    ]
    predictions_line_modified = [
        [set(_extract_modified_lines(diff)) for diff in row_diffs]
        for row_diffs in prediction_diffs
    ]
    # getting all the scores of each diff with the references
    all_pass_scores = [
        [100*_overlap(pred_diff,reference_row_diff) for pred_diff in prediction_row_diffs]
        for prediction_row_diffs,reference_row_diff in zip(
             predictions_line_modified,reference_line_modified
        )
    ]


    # getting the max score that the predictions achieved
    return [pass_score if pass_score else [0.0] for pass_score in all_pass_scores]
