from typing import List
import re


def compute_line_score(
    prediction_diffs: List[List[str]], reference_diffs: List[List[str]]
) -> list[float]:
    def _extract_modified_lines(diff_text: str) -> list[int]:
        # Regular expression to match the diff header for line numbers
        pattern = r"@@ -(\d+) \+\d+ @@"
        matches = re.findall(pattern, diff_text)
        # Convert matches to integers (line numbers)
        modified_lines = [int(line) for line in matches]
        return modified_lines

    def _max_overlap(prediction: set[int], references: List[set[int]]) -> float:
        # getting the biggest score from the comparison of the diff with the references diffs
        return max(
            [len(prediction & reference) / len(reference) for reference in references]
        )


    # computing the line score(i.e. the lines' numbers)
    ## for each reference and prediction diffs get a list of the lines modified
    reference_line_modified = [
        [set(_extract_modified_lines(diff)) for diff in row_diffs]
        for row_diffs in reference_diffs
    ]
    predictions_line_modified = [
        [set(_extract_modified_lines(diff)) for diff in row_diffs]
        for row_diffs in prediction_diffs
    ]
    # getting all the scores of each diff with the references
    all_pass_scores = [
        [_max_overlap(ref_dif, prediction_row_diffs) for ref_dif in reference_row_diffs]
        for reference_row_diffs, prediction_row_diffs in zip(
            reference_line_modified, predictions_line_modified
        )
    ]

    # getting the max score that the predictions achieved
    return [max(pass_score) if pass_score else 0.0 for pass_score in all_pass_scores]
