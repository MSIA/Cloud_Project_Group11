import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def feature_eng(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Implement feature engineering for model training, including one-hot encoding, dropping columns, and log transformation

    Args:
    - df (pd.DataFrame): Input data as a pandas DataFrame
    - config (dict): configuration for the feature_eng function
    """
    logger.info("Implementing feature engineering for modeling.")

    df = df.copy()
    # one hot encoding:
    if "OneHotEncoding" in config:   # if people want to implement one hot encoding
        for col in config["OneHotEncoding"]:
            # genre is a special column because one song could have multiple genras at the same time
            if col == "genre":  
                df_grouped = df.groupby("track_id")["genre"].agg(list).reset_index()
                # Create dummy variables for each “genre” value
                dummy_vars = pd.get_dummies(df_grouped["genre"].apply(pd.Series).stack()).groupby(level=0).sum()
                # Join the dummy variables with the original dataframe
                df_with_dummies = pd.concat([df_grouped, dummy_vars], axis=1)
                df_sub = df_with_dummies.drop(["genre"], axis = 1) # drop the original genre colum
                df = pd.merge(df_sub, df.drop_duplicates("track_id"), on="track_id").drop(["genre"], axis =1 )
                logger.debug("The genre column is one-hot encoded")

                try: 
                    assert("genre" not in list(df.columns))
                    logger.debug("The original genre is dropped")

                except AssertionError:
                    logger.error("The original genre is not dropped successfully")
                    sys.exit(1)

            else:
                try:
                    one_hot = pd.get_dummies(df[col])   # one hot encode the column and concat it to the df
                    df = pd.concat([df,one_hot], axis = 1) 
                    logger.debug("The %s column is one-hot encoded", col)

                    df.drop([col], axis = 1, inplace=True)  # drop the original column
                    assert(col not in list(df.columns)) # check the original column does not exist anymore
                    logger.debug("The original %s column is dropped", col)

                except KeyError:
                    logger.error("The %s column does not exist in the dataframe and thus cannot be one-hot encoded", col)
                    sys.exit(1)
                except AssertionError:
                    logger.error("The original %s column is not dropped successfully", col)
                    sys.exit(1)
    
    # drop columns
    if "columns_to_drop" in config:
        before_drop = len(list(df.columns)) # number of columns before dropping 
        for col in config["columns_to_drop"]:
            try:
                df.drop(col, axis = 1, inplace=True)
                logger.debug("Dropping %s column", col)
            except KeyError:
                logger.error("The column %s to be dropped is not found in the dataframe. Please provide a valid column", col)
                sys.exit(1)
        try: 
            # check the number of columns dropped matches with the number of columns in config["columns_to_drop"]
            assert((before_drop - len(df.columns)) == len(config["columns_to_drop"]))
            dropped_cols = str(config["columns_to_drop"])
            logger.debug("Columns are successfullt dropped as required. Drooped columns: "+dropped_cols )
        except AssertionError:
            logger.error("Not all columns are dropped successfully")
            sys.exit(1)

    # log transform
    if "log_transform" in config:
        for out_col, in_col in config["log_transform"].items():
            try: 
                df[out_col] = df[in_col].apply(lambda x: x+200).apply(np.log)
                logger.debug("The column %s is successfully logged, and a new column %s is generated"% (in_col, out_col))
                df.drop(in_col, axis = 1, inplace=True)
                assert(in_col not in list(df.columns))
                logger.debug("The original %s column is successfully dropped", in_col)
            except KeyError:
                logger.error("The column %s to be logged is not found in the dataframe. Please provide a valid column", in_col)
                sys.exit(1)
            except TypeError:
                logger.error("The column %s has wrong data type. Please provide a numeric column", in_col)
                sys.exit(1)
            except AssertionError:
                logger.error("The original column %s is not dropped successfully")
    
    keys = list(config.keys())
    for k in keys:
        try: 
            if k not in ["OneHotEncoding", "columns_to_drop", "log_transform"]:
                raise ValueError
        except ValueError:
            logger.error("The operation %s is out of scope. Please provide a valid feature engineering technique "
                         + "amomg OneHotEncoding, columns_to_drop, and log_transform", k)
            
    # drop any rows with missing values
    df.dropna(inplace=True)
    logger.info("Feature engineering finished")

    return df



        


