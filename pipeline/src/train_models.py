"""
This module provides functionality for training regression models using various algorithms
like RandomForest, Neural Networks, and XGBoost. It also provides functionality for hyperparameter
tuning using GridSearchCV.
"""
import logging
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


logger = logging.getLogger(__name__)

def split_data(dataset: pd.DataFrame, config: dict):
    """
    Splits the given dataset into train and test datasets.
    
    Args:
        dataset(dataframe): Input dataset ready to be split into train and test
        config(dict): Configuration for train_model function

    Returns:
        x_train, x_test, y_train, y_test(dataframes): Splits of the input dataset
        target_column(str): Name of the target column
    """
    try:
        logger.info("Splitting the dataset into train and test")
        target_column = config["target_column"]
        target = dataset[target_column]
        features = dataset.drop(target_column, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, **config["train_test_split"]
        )
    except KeyError:
        logger.error("Please make sure the config is correct \
                     and the target column exists in the dataset")
        raise
    except Exception:
        logger.error("Error encountered")
        raise
    return x_train, x_test, y_train, y_test, target_column

def perform_grid_search(x_train, y_train, config: dict):
    """
    Performs grid search for the specified models and their hyperparameters.
    
    Args:
        x_train(dataframe): Training data
        y_train(dataframe): Target data
        config(dict): Configuration for models and their hyperparameters

    Returns:
        result(list): List of fitted GridSearchCV objects, their best scores and model names
    """
    model_mapping = {
        "random_forest": RandomForestRegressor(),
        "neural_network": MLPRegressor(),
        "XGboost": xgb.XGBRegressor()
    }

    # Initialize result list
    result = []

    # Implement grid search for each model in config
    try:
        logger.info("Performing grid search for models")
        # Implement grid search for each model in config
        for model_type, model_params in config["model"].items():
            if model_type in model_mapping:
                model = model_mapping[model_type]
                grid_search = GridSearchCV(
                    model, model_params, cv=3, verbose=1, n_jobs=-1, scoring="r2"
                )
                grid_search.fit(x_train, y_train)
                result.append([grid_search, grid_search.best_score_, model_type])
    except Exception as err:
        logger.error("Error encountered during grid search: %s",err)
        raise
    return result

def assign_models(result):
    """
    Extracts the best model, its name, and other models from the results of grid search.
    
    Args:
        result(list): List of fitted GridSearchCV objects, their best scores and model names

    Returns:
        best_model(sklearn.base.BaseEstimator): Best fitted model
        best_model_name(str): Name of the best model
        other_models(dict): Dictionary of other models and their names
    """
    try:
        logger.info("Assigning best and other models")
        best_model, best_model_name, other_models = result[0][0].best_estimator_, \
            result[0][2], {x[2]:x[0].best_estimator_ for x in result[1:]}
    except IndexError:
        logger.error("Please make sure the result list is not empty")
        raise
    except Exception:
        logger.error("Error encountered while assigning models")
        raise
    return best_model, best_model_name, other_models

def append_target(df_list, target_list, target_column):
    """
    Appends the target column to the given dataframes.
    
    Args:
        df_list(list): List of dataframes
        target_list(list): List of target data
        target_column(str): Name of the target column

    Returns:
        df_list(list): List of dataframes with target column appended
    """
    try:
        logger.info("Appending target column to dataframes")
        for data_frame, target_data in zip(df_list, target_list):
            data_frame[target_column] = target_data
    except Exception:
        logger.error("Error encountered while appending target column")
        raise
    return df_list

# training models
def train_model(
    dataset: pd.DataFrame, config: dict) -> tuple[RandomForestRegressor,
                                                  MLPRegressor, xgb.XGBModel,
                                                  pd.DataFrame]:
    """
    Orchestrator function to split the data, perform grid search, \
        assign models, and append the target column.
    
    Args:
        dataset(dataframe): Input dataset ready to be split into train and test
        config(dict): Configuration for train_model function

    Returns:
        Best trained model instance
        Name of the best model
        Dictionary with other models
        Training dataframe including all features
        Test dataframe including all features
    """
    logger.info("Begine to train and tune hyperparameters for models")
    try:
        x_train, x_test, y_train, y_test, target_column = split_data(dataset, config)
        result = perform_grid_search(x_train, y_train, config)
        best_model, best_model_name, other_models = assign_models(result)
        x_train, x_test = append_target([x_train, x_test], [y_train, y_test], target_column)
    except Exception as err:
        logger.error("Error encountered in train_model: %s", err)
        raise
    return best_model, best_model_name, other_models, x_train, x_test

def save_data(train: pd.DataFrame, test: pd.DataFrame, artifacts: Path):
    """
    Saves the training and test data to disk.

    Args:
        train: A Pandas DataFrame containing the training data.
        test: A Pandas DataFrame containing the test data.
        artifacts: A Path object specifying the directory to save the data to.
    """
    logger.info("Saving the train and test dataframe")

    try:
        train.to_csv(artifacts / "train.csv", index=False)
        test.to_csv(artifacts / "test.csv", index=False)
        logger.info("Training and test data saved to disk.")
    except FileNotFoundError as err:
        logger.error(
            "Please provide a valid path to store the train and test data: %s", err
        )

    logger.info("Finished saving the train and test dataframe")

def save_model(model, file_path):
    """
    Saves the trained machine learning model to disk.

    Args:
        model: The trained machine learning model.
        artifacts: A Path object specifying the directory to save the model to.
    """
    logger.info("Saving the trained model into a pickle file")
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.info("The best trained model saved to disk.")
    except FileNotFoundError as error:
        logger.error("File path not found. Please provide a \
                     valid path to store the model: %s", error)

    logger.info("Finished saving trained model to disk.")
