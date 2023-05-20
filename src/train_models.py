import sys
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

def train_model(
    dataset: pd.DataFrame, config: dict
) -> tuple[RandomForestRegressor, MLPRegressor, xgb.XGBModel, pd.DataFrame]:
    """
    Splits into train and test data sets.  
    Select and train the best model among RandomForest, Neural Network, 
    and XGboost based on the result of Gridsearch. 

    Args:
        dataset(dataframe): Input dataset ready to be split into train and test
        config(dict): Configuration for train_model function

    Returns:
        Trained best model instance
        Training dataframe including all features
        Test dataframe including all features
    """
    logger.info("Begine to train and tune hyperparameters for models")

    # Split the dataset into train and test
    target_column = config["target_column"]
    try:
        target = dataset[target_column]
        features = dataset.drop(target_column, axis=1)
    except KeyError as e:
        logger.error("The target column name provided is not found  %s", e)
        sys.exit(1)


    X_train, X_test, y_train, y_test = train_test_split(
        features, target, **config["train_test_split"]
    )
 
    logger.debug("Train test data is successfully splitted")

    result = []
    # Implement grid search for random forest, neural network, and xgboost
    try:
        for model_type in config["model"]:
            # hyperparamter options for each model type
            params = config["model"][model_type]

            # Random Forest Grid Search
            if model_type == "random_forest":
                rf =  RandomForestRegressor()
                grid_search_rf = GridSearchCV(rf, params, cv=5, verbose=1, n_jobs=-1, scoring="r2")
                grid_search_rf.fit(X_train, y_train)
                rf_score = grid_search_rf.best_score_
                # Store the best RF model in the result list
                rf_result = [grid_search_rf, rf_score, model_type]
                result.append(rf_result)

            # Neural Network Grid Search
            elif model_type == "neural_network":
                nn = MLPRegressor()
                grid_search_nn = GridSearchCV(nn, params, cv=5, verbose=1, n_jobs=-1, scoring="r2")
                grid_search_nn.fit(X_train, y_train)
                nn_score = grid_search_nn.best_score_
                # Store the best NN model in the result list
                nn_result = [grid_search_nn, nn_score, model_type]
                result.append(nn_result)

            # XGBoost model Grid Search
            elif model_type == "XGboost":
                xgboost = xgb.XGBRegressor()
                grid_search_xgb = GridSearchCV(xgboost, params, verbose =1, scoring='r2')
                grid_search_xgb.fit(X_train, y_train)
                xgb_score = grid_search_xgb.best_score_
                # Store the best NN model in the result list
                xgb_result = [grid_search_xgb, xgb_score, model_type]
                result.append(xgb_result)

            # other model type options are out of scope in this project
            else:
                logger.error("The model type %s is out of scope. " +
                             "Available model types are \'random_forest\', \'neural_network\', and \'XGboos\''", model_type)
                sys.exit(1)

    except KeyError: #TODO: change this error name later
            logger.error("Failed to implement grid search for random forest, nueral network, or XGboost model.")
            sys.exit(1)

    # Select the best model
    result.sort(key=lambda x: x[1], reverse=True)
    # best model object from gid search
    best_gridsearch = result[0][0]
    best_model = best_gridsearch.best_estimator_
    best_model_name = result[0][2]
    logger.debug("The best model with higest r2 score from the grid search is %s",  best_model_name)

    train_df = X_train.copy()
    train_df[target_column] = y_train
    test_df = X_test.copy()
    test_df[target_column] = y_test

    logger.info("Finished training the model.")

    return best_model, train_df, test_df
        

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
    except FileNotFoundError as e:
        logger.error(
            "Please provide a valid path to store the train and test data: %s", e
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
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("The best trained model saved to disk.")
    except FileNotFoundError as e:
        logger.error("File path not found. Please provide a valid path to store the model: %s", e)
    except Exception as e:
        logger.exception("Error occurred while saving the model: %s", e)

    logger.info("Finished saving trained model to disk.")