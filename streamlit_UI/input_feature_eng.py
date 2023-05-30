"""
This module is build for OneHot Encoding users' input 
"""
# pylint: disable=invalid-name
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger("streamlit-feature-engineer")

def oneHot_input(input_df,colname) -> pd.DataFrame:
    """
    This function one-hot encodes one categorical column at a time

    Args: 
        input_df: user input df
        all values: a list all possible values of that column in your training data
        colname: column that you want to oneHot Encode
        return a dataframe which the column is one hot encoded
    """

    all_values = {
                "genre": ['A Capella', 'Alternative', 'Anime', 'Blues', 'Children’s Music',
                            'Classical', 'Comedy', 'Country', 'Dance', 'Electronic', 'Folk',
                            'Hip-Hop', 'Indie', 'Jazz', 'Movie', 'Opera', 'Pop', 'R&B', 'Rap',
                            'Reggae', 'Reggaeton', 'Rock', 'Ska', 'Soul', 'Soundtrack', 'World'],
                "time_sig": ['1/4', '3/4', '4/4','5/4'],
                "mode": ['Major', 'Minor'],
                "key": ['A', 'A#', 'B','C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
                }


    all_values_df = pd.DataFrame({colname: all_values[colname]})
    train_dummies = pd.get_dummies(all_values_df, columns=[colname],prefix='', prefix_sep='')
    # One Hot encode the test data in the same way as for training data
    test_dummies = pd.get_dummies(input_df, columns=[colname],prefix='', prefix_sep='')
    # Make sure the test data has the same columns as the training data
    test_dummies = test_dummies.reindex(columns = train_dummies.columns, fill_value=0)

    return test_dummies



def input_feature_eng(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Implement feature engineering for model training, 
    including one-hot encoding, dropping columns, and log transformation

    Args:
    - df (pd.DataFrame): Input data as a pandas DataFrame
    - config (dict): configuration for the feature_eng function
    """

    df = df.copy()
    # result_df = pd.DataFrame()
    # one hot encoding:
    if "OneHotEncoding" in config:   # if people want to implement one hot encoding
        for col in config["OneHotEncoding"]:
            # genre is a special column because one song could have multiple genras at the same time
            col_dummies = oneHot_input(df,col)
            df = pd.concat([col_dummies,df], axis = 1)
            logger.debug("The column %s is successfully one-hot encoded")
            # drop original column
            df.drop(col, axis =1, inplace=True)

    # log transform
    if "log_transform" in config:
        for out_col, in_col in config["log_transform"].items():
            df[out_col] = df[in_col].apply(lambda x: x+200).apply(np.log)
            logger.debug(("The column %s is successfully logged, \
                          and a new column %s is generated", (in_col, out_col)))
            df.drop(in_col, axis = 1, inplace=True)
            logger.debug("The original %s column is successfully dropped", in_col)

    # drop any rows with missing values
    df.dropna(inplace=True)
    logger.info("Feature engineering finished")

    # reorder the columns so that it matches with
    # Desired order of columns
    desired_order = ['A Capella', 'Alternative', 'Anime', 'Blues', 'Children’s Music',
       'Classical', 'Comedy', 'Country', 'Dance', 'Electronic', 'Folk',
       'Hip-Hop', 'Indie', 'Jazz', 'Movie', 'Opera', 'Pop', 'R&B', 'Rap',
       'Reggae', 'Reggaeton', 'Rock', 'Ska', 'Soul', 'Soundtrack', 'World',
       'acousticness', 'danceability', 'energy', 'valence', 'A', 'A#', 'B',
       'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', '1/4', '3/4', '4/4',
       '5/4', 'Major', 'Minor', 'log_duration', 'log_instrumentalness',
       'log_liviness', 'log_loudness', 'log_speechness', 'log_tempo']

    # Reorder the columns
    reordered_data = df.reindex(columns=desired_order)

    return reordered_data
