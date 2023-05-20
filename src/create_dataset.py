import sys
import logging
from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)

def save_dataset(data: pd.DataFrame, output_file: str) -> None:
    """
    Save structured dataset to disk.

    Args:
    - data (pd.DataFrame): Input data as a pandas DataFrame
    - output_file (str): path to output file.
    """
    logger.info("Saving the dataframe to the disk.")
    try: 
        data.to_csv(output_file, index=False)
        logger.info("The data frame is succeffully saved to disk")
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save output file to.")
        sys.exit(1)


def create_cleaned_dataset(data_path: Path) -> pd.DataFrame:
    """
    Read in the structured csv data and clean the dataset

    Args:
    - data_path (Path): path to the csv file.
    - columns: a list of column names for the created dataset.
    """

    logger.info("Creating and cleaning the dataset")

    # read in dataframe
    try:
        df = pd.read_csv(data_path)
        logger.debug("The data is successfully read in")
    except FileNotFoundError:
        logger.error("Please provide a valid file location to read in the csv file.")
        sys.exit(1)

    # drop rows with time signature = 0/4
    cleaned_data = df.copy()  # Create a copy of the DataFrame
    try:
        cleaned_data = cleaned_data[cleaned_data["time_signature"] != '0/4']
        logger.debug("Time_Signature = 0/4 is removed from the dataframe")
    except KeyError:
        logger.error("time_signature is not found in the dataframe, so data cleaning for time_signature failed")
        sys.exit(1)

    # Merge repetitive Children's Music genre into a single category:
    music1 = "Children's Music" # first children's music genre name
    music2 = "Children’s Music" # second children's music genre name
    cleaned_data["genre"].replace(music1 , music2, inplace=True)
    logger.debug("Megered repetitive Children's Music genre into a single category")

    # drop rows with missing values
    cleaned_data = cleaned_data.dropna()
    logger.debug("Successfully dropped NA in the dataframe")

    logger.info("The dataframe is successfully created and cleaned")
    return cleaned_data
