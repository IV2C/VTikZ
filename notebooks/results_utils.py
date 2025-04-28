import math
import pandas as pd
from datasets import Dataset
from datasets.formatting.formatting import LazyRow

class MetricPolicy:
    @staticmethod
    def mathematical_average(values: list[float], weights: list[float] = None) -> float:
        if weights is None:
            return sum(values) / len(values)
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    @staticmethod
    def geometrical_average(values: list[float], weights: list[float] = None) -> float:
        if weights is None:
            return math.prod(values) ** (1 / len(values))
        total_weight = sum(weights)
        return math.prod(v ** (w / total_weight) for v, w in zip(values, weights))

    @staticmethod
    def harmonic_mean(values: list[float]) -> float:
        return len(values) / sum(1 / v for v in values)

    @staticmethod
    def metric_priority(df: pd.DataFrame, metric_priority_order: list[str]) -> float:
        df = df.sort_values(metric_priority_order,ascending=False)
        print(df)
        pass

    @staticmethod
    def compute_best_prediction(row: LazyRow):
        """
        Computes the best prediction out of arrays of metrics, according to a policy(arithmetic, geometrical, or harmonic mean)

        Args:
            row (_type_): the row to make the treatment on
            metrics (list[Metric]): the list of metrics to compute the best prediction on

        """

        print(type(row))
        
        
        computed_metrics_names = [
            metric_name
            for metric_name in row.keys
            if metric_name.endswith("Metric")
        ]

        if row["compiling_score"] == 0:
            # nothing was able to be computed from the predictions
            row["index_best_prediction"] = -1
            for metric_name in computed_metrics_names:
                row[f"best_{metric_name}"] = 0
            return row
        scores_predictions_array = []
        df_scores = row.to_pandas().explode(computed_metrics_names)

        # TODO Update with metric_analysis output
        most_important_metrics = ["PatchMetric", "LineMetric"]
        metric_priority_order = most_important_metrics + (
            computed_metrics_names - most_important_metrics
        )

        policy_applied_array = [
            (
                MetricPolicy.metric_priority(df_scores, metric_priority_order)
                if None not in current_scores
                else float("-inf")
            )
            for current_scores in scores_predictions_array
        ]

        max_value = max(policy_applied_array)
        index_max_value = policy_applied_array.index(max_value)
        row["index_best_prediction"] = index_max_value
        for metric_name in computed_metrics_names:
            row[f"best_{metric_name}"] = row[metric_name][index_max_value]
        return row


import pandas as pd


def flatten_metrics(df, metric_names):
    expanded_rows = []

    other_columns = df.columns.difference(metric_names)

    for _, row in df.iterrows():
        metric_series = row[metric_names]
        len_metrics = len(metric_series.iloc[0])
        for i in range(len_metrics):
            new_row = []
            for metric in metric_series:
                new_row.append(metric[i])
            new_row.extend(row[other_columns])
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows, columns=list(metric_names) + list(other_columns))
