import math


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

    def compute_best_prediction(row, computed_metrics_names: list[str]):
        """
        Deprecated - now handled via the jupyter notebook

        Computes the best prediction out of arrays of metrics, according to a policy(arithmetic, geometrical, or harmonic mean)

        Args:
            row (_type_): the row to make the treatment on
            metrics (list[Metric]): the list of metrics to compute the best prediction on

        """
        if row["compiling_score"] == 0:
            # nothing was able to be computed from the predictions
            row["var_score"] = 0
            row["index_best_prediction"] = -1
            for metric_name in computed_metrics_names:
                row[f"best_{metric_name}"] = 0
            return row
        scores_predictions_array = []
        computed_metric_amount = len(
            row[computed_metrics_names[0]]
        )  # assuming all metrics computed the same amount of scores
        for i in range(computed_metric_amount):
            current_score_array = []
            for metric_name in computed_metrics_names:
                current_score_array.append(row[metric_name][i])
            scores_predictions_array.append(current_score_array)

        policy_applied_array = [
            MetricPolicy.mathematical_average(current_scores)
            for current_scores in scores_predictions_array
        ]

        max_value = max(policy_applied_array)
        index_max_value = policy_applied_array.index(max_value)
        row["var_score"] = max_value
        row["index_best_prediction"] = index_max_value
        for metric_name in computed_metrics_names:
            row[f"best_{metric_name}"] = row[metric_name][index_max_value]
        return row

import pandas as pd
def flatten_metrics(df,metric_names):
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


    return pd.DataFrame(expanded_rows, columns=list(metric_names)+list(other_columns))