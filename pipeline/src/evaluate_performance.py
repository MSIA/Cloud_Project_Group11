"""
This module contains functions for evaluating the performance of a model and saving the metrics.
"""

import sys
import logging
from typing import Dict
from pathlib import Path
import yaml
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)


def evaluate_performance(scores: pd.DataFrame, config: Dict) -> Dict:
    """
    Evaluate the performance of the model using the scores and configuration.

    Parameters:
    scores (DataFrame): A dataframe containing the scores generated from the model.
    config (Dict[str, Any]): A dictionary containing the configuration parameters.

    Returns:
    Dict[str, Any]: A dictionary containing the metrics computed from the scores.
    """
    logger.info("Evaluating the performance.")

    evaluation = {}
    try:
        y_test = scores["True value"]
        y_pred = scores["prediction"]

        # Compute all possible metrics for classification problem
        for metric in config["metrics"]:
            metric = metric.lower()
            if metric == "r2":
                r2_score_value = r2_score(y_test, y_pred)
                evaluation["R2"] = float(r2_score_value)
            elif metric == "mse":
                mse = mean_squared_error(y_test, y_pred)
                evaluation["MSE"] = float(mse)
            elif metric == "mae":
                mae = mean_absolute_error(y_test, y_pred)
                evaluation["MAE"] = float(mae)
            else:
                raise ValueError
    except ValueError:
        logger.error("Unknown metric: %s", metric)
        sys.exit(1)
    except KeyError as key_error:
        logger.error("The key %s is not found in the scores DataFrame or config dictionary.",
                      str(key_error))
        sys.exit(1)

    logger.info("Performance evaluation successful.")
    return evaluation


def save_metrics(metrics: Dict, filepath: Path) -> None:
    """
    Save the metrics to a YAML file.

    Parameters:
    metrics (Dict[str, Any]): A dictionary containing the metrics computed from the scores.
    filename (str): The name of the YAML file to save the metrics to.
    """
    logger.info("Saving the evaluation metrics")
    try:
        with open(filepath, "w", encoding='utf-8') as file:
            yaml.dump(metrics, file, default_flow_style=False)

        logger.info("Metrics saved successfully.")

    except Exception as error:
        logger.error("An error occurred while saving the metrics.")
        logger.error(str(error))
        raise error
