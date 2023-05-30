"""
This module automates the pipeline of training models and evaluating their performance.
It reads configuration and data from an AWS S3 bucket, trains models as per the configuration,
saves the models and their evaluations in a local artifacts directory and finally uploads the
artifacts to an output bucket in S3.
"""
# pylint: disable=too-many-locals
import json
import datetime
from time import sleep
from io import StringIO
import logging.config
import os
from pathlib import Path
import typer
import pandas as pd
import yaml
import boto3

# import src.analysis as eda
import src.evaluate_performance as ep
import src.score_model as sm
import src.train_models as tm
import src.aws_utils as aws


logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("main")

# Set up output directory for saving artifacts
now = int(datetime.datetime.now().timestamp())
artifacts = Path("artifacts") / str(now)
artifacts.mkdir(parents=True)

BUCKET_NAME_RAW = 'msia423-group11-spotifty'


# download the yaml file from S3 "msia423-group11-spotifty"
def load_config(config_ref: str, bucket_name: str) -> dict:
    """
    Load a configuration file from an S3 bucket.

    Args:
    config_ref : str
        The key of the configuration file in the bucket.
    bucket_name : str
        The name of the bucket containing the configuration file.

    Returns:
    dict
        A dictionary containing the configuration parameters.
    """
    try:
        s_3 = boto3.client('s3')
        response = s_3.get_object(Bucket=bucket_name, Key=config_ref)
        config = yaml.load(response['Body'], Loader=yaml.SafeLoader)
        return config
    except yaml.error.YAMLError:
        logger.error("Error while loading configuration from %s", bucket_name)
        return {}
    logger.info("Configuration file loaded from %s", bucket_name)


def run_pipeline(config,input_key, input_bucket):
    """
    Execute the data pipeline for model training and evaluation.

    Args:
    config : dict
        The configuration parameters for the pipeline.
    input_key : str
        The key of the input data in the S3 bucket.
    input_bucket : str
        The name of the S3 bucket containing the input data.
    """
    # load s3 for ready data

    bucket_name_ready_to_train = input_bucket
    obj_key = input_key

    # get the name of output bucket
    bucket_name_artifacts = config["aws"]["output_bucket"]

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as file:
        yaml.dump(config, file)

    # Load data ready to train from S3
    try:
        s_3 = boto3.client('s3')
        csv_obj = s_3.get_object(Bucket=bucket_name_ready_to_train, Key=obj_key)
        csv_string = csv_obj['Body'].read().decode('utf-8')
        data = pd.read_csv(StringIO(csv_string))
        logger.debug("The data is successfully read in")
    except FileNotFoundError:
        logger.error("The %s is not exist within the given S3 bucket",obj_key)

    # Split data into train/test set and train model based on config; save each to disk
    best_model, best_model_name, othermodels, \
        train, test = tm.train_model(data, config["train_model"])

    # Save all models to the artifacts folder and splitted data
    model_path = artifacts / "models"
    os.makedirs(model_path)
    model_name = f"{best_model_name}_best.pkl"
    tm.save_model(best_model, model_path / model_name)

    for key, value in othermodels.items():
        model_name = f'{key}.pkl'
        tm.save_model(value, model_path /model_name)
    tm.save_data(train, test, artifacts)

    # Model Evaliation:
    # Create a folder to store performance evaluations for all models
    evaluation_path = artifacts / "model_evaluations"
    os.makedirs(evaluation_path)
    # Loop through all model pickle files in the models folder
    for filename in os.listdir(model_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(model_path, filename)
            # make predictions on test set for all models; save scores(predictions) to disk
            scores = sm.model_prediction(test, file_path , config["score_model"])
            score_file = filename.split('.')[0] + "_scores.csv"
            sm.save_scores(scores, evaluation_path / score_file)

            # Evaluate performances for all models in the folder; save metric files to disk
            metrics = ep.evaluate_performance(scores, config["evaluate_performance"])
            metric_file = filename.split('.')[0] + "_metrics.yaml"
            ep.save_metrics(metrics, evaluation_path / metric_file)

    # Save metrics to S3
    aws.upload_artifacts(artifacts, bucket_name_artifacts, "artifacts")

# load data from
def process_message(msg: aws.Message):
    """
    Process a message from an SQS queue.

    Args:
    msg : aws.Message
        The message to process.
    """
    message_body = json.loads(msg.body)
    bucket_name = message_body["detail"]["bucket"]["name"]
    object_key = message_body["detail"]["object"]["key"]
    config = load_config("default-config.yaml",BUCKET_NAME_RAW)
    if object_key == "ready_data.csv":
        logger.info("Running pipeline with data from: %s", bucket_name)
        run_pipeline(config,input_bucket=bucket_name,input_key=object_key)

def main(
    sqs_queue_url: str,
    max_empty_receives: int = 3,
    delay_seconds: int = 10,
    wait_time_seconds: int = 10):
    """
    The main function of the script. It continuously polls an SQS queue for messages
    and processes each message until a specified number of empty responses have been received.

    Args:
    sqs_queue_url : str
        The URL of the SQS queue.
    max_empty_receives : int, optional
        The maximum number of empty responses to receive from the queue 
        before stopping (default is 3).
    delay_seconds : int, optional
        The duration (in seconds) to sleep after each request to the queue (default is 10).
    wait_time_seconds : int, optional
        The duration (in seconds) for which the call will wait for a message to arrive in the
        queue before returning (default is 10).
    """
    # Keep track of the number of times we ask queue for messages but receive none
    empty_receives = 0
    # After so many empty receives, we will stop processing and await the next trigger
    while empty_receives < max_empty_receives:
        logger.info("Polling queue for messages...")
        messages = aws.get_messages(
            sqs_queue_url,
            max_messages=2,
            wait_time_seconds=wait_time_seconds,
        )
        logger.info("Received %d messages from queue", len(messages))

        if len(messages) == 0:
            # Increment our empty receive count by one if no messages come back
            empty_receives += 1
            sleep(delay_seconds)
            continue

        # Reset empty receive count if we get messages back
        empty_receives = 0
        for message in messages:
            # Perform work based on message content
            try:
                process_message(message)
            # We want to suppress all errors so that we can continue processing next message
            except ValueError as err:
                logger.error("Unable to process message, continuing...")
                logger.error(err)
                continue
            except TypeError as type_err:
                logger.error("Unable to process message due to a TypeError: %s", {type_err})
                continue
            # We must explicitly delete the message after processing it
            aws.delete_message(sqs_queue_url, message.handle)
        # Pause before asking the queue for more messages
        sleep(delay_seconds)


if __name__ == "__main__":
    typer.run(main)
