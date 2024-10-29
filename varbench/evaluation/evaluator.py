import difflib
from datasets import Dataset
import subprocess
import os
from ..prompt_templates import *
from ..model import LLM_Model
from loguru import logger


def evaluate(subset: Dataset, model: LLM_Model):

    subset_processed = subset.map(create_message_row)

    logger.info(subset_processed["messages"])

    predictions = model.batchRequest(
        subset_processed["messages"], subset_processed["id"]
    )
    subset_processed: Dataset = subset_processed.add_column("predictions", predictions)
    subset_processed.save_to_disk(".tmp/computed_dataset_" + model.model_name)

    return _compute(
        subset_processed["id"],
        subset_processed["code"],
        subset_processed["predictions"],
        subset_processed["diffs"],
    )


def create_message_row(row):

    user_instruction = IT_PROMPT.format(
        instruction=row["instruction"], content=row["code"]
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {"role": "user", "content": user_instruction},
    ]

    row["messages"] = messages
    return row


def _compute(ids, inputs, predictions, diffs):

    def _diffs(input, predictions: list) -> list:
        return [
            "".join(list(difflib.unified_diff(input, prediction, n=0))[2:])
            for prediction in predictions
        ]

    """Returns the scores"""
    result_list = [
        bool(set(_diffs(i, p)) & set(d)) for i, p, d in zip(inputs, predictions, diffs)
    ]

    individual_scores = {id: result for id, result in zip(ids, result_list)}

    varscore = sum(result_list) / len(predictions)
    return {"varscore": varscore, "individual_scores": individual_scores}
