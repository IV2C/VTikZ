## Importing dataset from huggingface
from datasets import (
    Dataset,
    load_dataset,
    concatenate_datasets,
    get_dataset_config_names,
)
import pandas as pd

all_configs = [
    config_name
    for config_name in get_dataset_config_names("CharlyR/vtikz-evaluation")
    if "benchmark" in config_name
]

all_datasets: list[Dataset] = []

for config in all_configs:
    conf_ds = load_dataset("CharlyR/vtikz-evaluation", config, split="tikz")
    config_name_column = ["".join(config.split("_benchmark"))] * len(conf_ds)
    all_datasets.append(conf_ds.add_column("config", config_name_column))


dataset = concatenate_datasets(all_datasets)
df = dataset.to_pandas()[["config", "id", "parsing_score", "compiling_score"]]

#### Filtering and organizing dataset

computed_metrics_names = [
    metric_name
    for metric_name in dataset.column_names
    if metric_name.endswith("Metric")
]

most_important_metrics = [
    "TemplateMetric",
    "ImageEqualityMetric",
    "LineMetric",
    "CrystalBleuPatchMetric",
]
metric_priority_order = most_important_metrics + (
    list(set(computed_metrics_names) - set(most_important_metrics))
)

### computing best metrics
result_df: pd.DataFrame = dataset.to_pandas().explode(
    computed_metrics_names
    + ["patch", "template_solution_code", "code_solution", "difficulty_ast"]
)
result_df: pd.DataFrame = result_df.explode(
    computed_metrics_names + ["images_result", "predictions", "predictions_patches"]
)
df_sorted = result_df.sort_values(by=metric_priority_order, ascending=False)
df_sorted

# Get the highest row per group based on sorting order
result = df_sorted.groupby(["id", "config"]).first().reset_index()


result_df = result[computed_metrics_names + ["id", "config", "difficulty"]].astype(
    {m_name: "float" for m_name in computed_metrics_names}
)

filtered_df = result_df[metric_priority_order + ["id", "config", "difficulty"]]

# computing metrics
import numpy as np

filtered_df["SuccessCustomizationMetric"] = np.where(
    (filtered_df["TemplateMetric"] == 100)
    | (filtered_df["ImageEqualityMetric"] == 100),
    100,
    0,
)
filtered_df["LocationMetric"] = np.where(filtered_df["LineMetric"] >= 100, 100, 0)


## Removing unwanted models
filtered_df = filtered_df[
    (~filtered_df["config"].str.contains("VIF")) &
    (~filtered_df["config"].str.contains("FAR"))
]


#### Renaming models
model_name_map = {
    "llama3.18binstant": "Llama-3.1-8B",
    "mixtral8x7b32768": "Mixtral-8x7B",
    "llama370b8192": "Llama-3-70B",
    "deepseekr1distillllama70b": "DeepSeek-R1-Distill-Llama-70B",
    "llama3.370bversatile": "Llama-3.3-70B",
    "gpt4o20240806": "GPT-4o-2024-08-06",
    "gemini2.0flash": "gemini-2.0-flash",
    "metallamallama4scout17b16einstruct": "Llama-4-Scout-17B-16E",
}

name_map = {"LLM": "Text", "LMM": "Text+Image", "VIF": "T", "FAR": "FAR"}


def compute_class(row):
    row["total"] = 1
    row["Compile"] = not row.isnull().any()
    row["Location"] = row["LocationMetric"] == 100
    row["SuccessCustomization"] = row["SuccessCustomizationMetric"] == 100
    row["model"] = model_name_map[row["config"].split("_")[1]]
    row["Modality"] = name_map[row["config"].split("_")[0].removeprefix("simple")]
    row["N"] = row["config"].split("_")[3]
    row["temp."] = row["config"].split("_")[5]
    row["version"] = row["config"].split("_")[-1]
    return row


computed_metrics_names = [
    metric_name for metric_name in filtered_df.columns if metric_name.endswith("Metric")
]


classed_dataset = filtered_df[
    computed_metrics_names + ["difficulty", "id", "config"]
].apply(compute_class, axis=1)[
    [
        "Modality",
        "N",
        "temp.",
        "version",
        "model",
        "total",
        "Compile",
        "Location",
        "SuccessCustomization",
    ]
]
classed_dataset = (
    classed_dataset.groupby(
        [
            "model",
            "Modality",
            "N",
            "temp.",
            "version"
        ]
    )
    .sum()
    .reset_index()
    .sort_values("SuccessCustomization")
)
grouped = classed_dataset.groupby("version")

# Convert to dictionary with version as keys and corresponding records as list of dicts
import json
data_by_version = {
    version: group.to_dict()
    for version, group in grouped
}

# Write to JSON
with open("leaderboard.json", "w") as f:
    json.dump(data_by_version, f, indent=2)
