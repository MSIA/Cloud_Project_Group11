import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import pickle


logger = logging.getLogger(__name__)

def model_prediction(
    test_data: pd.DataFrame, model_path: Path, model_config: Dict
) -> pd.DataFrame:
    """
    Generates predictions for the test data using the trained model object.

    Args:
        test_data (pd.DataFrame): DataFrame containing the test features and target.
        model_path (Path): Path of trained model pickle file
        model_config (Dict): Dictionary containing configuration for the model.

    Returns:
        pd.Dataframe: A dataframe of predicted probabilities for the target variable.
    """
    logger.info("Predicting on the test data")

    try:
        # Extract features from test data
        features = test_data.drop(model_config["target_column"], axis = 1)
        y_test = test_data[model_config["target_column"]]

        # load model from the pickle file
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Generate predictions
        ypred= model.predict(features)
        score_df = pd.DataFrame(
            {
                "prediction": ypred,
                "True value": y_test,
            }
        )
    except KeyError as e:
        logger.error("Invalid features for prediction on test data")
        raise e
    
    logger.info("Finished prediction")
    return score_df


def save_scores(scores: pd.DataFrame, filepath: Path) -> None:
    """
    Save the predicted probabilities and predictions to a CSV file.

    Args:
        scores (pd.DataFrame): A dataframe that stores predicted probabilities and predictions
        filepath (path): Path to save the scores CSV file.
    """
    logger.info("Saving the scores (prediction)")
    try:
        # Save the scores to a CSV file
        scores.to_csv(filepath, index=False)
        logger.info("Finished saving the scores (prediction)")
    except FileNotFoundError as e:
        logger.error("Please provide a valid filepath to save the scores csv ")
        raise e
