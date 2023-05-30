"""
This module provides utilities for interacting with AWS services like S3 and SQS.
It includes functions to upload artifacts to an S3 bucket and handle messages from an SQS queue.
"""
from pathlib import Path
import logging
from dataclasses import dataclass
import boto3 # type: ignore
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)

def upload_artifacts(artifacts: Path, bucket:str, prefix:str) -> None:
    """Upload all the artifacts in the specified directory to S3


    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        bucket: Bucket that want to upload to 
        key: the key that we save in the bucket

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    try:
        s3_client = boto3.client('s3')

        for artifact_path in artifacts.glob('**/*'):
            if artifact_path.is_file():
                s3_key = f'{prefix}/{artifact_path.relative_to(artifacts)}'
                try:
                    s3_client.upload_file(str(artifact_path), bucket, s3_key)
                except ClientError as err:
                    logging.error(err)
    except Exception as err:
        logger.error('Error while uploading artifacts to S3')
        raise err
    logger.info('Artifacts uploaded to S3 successfully')

@dataclass
class Message:
    """
    A class used to represent an SQS Message.

    Attributes:
    handle : str
        The receipt handle of the message.
    body : str
        The body content of the message.
    """
    handle: str
    body: str


def get_messages(
    queue_url: str,
    max_messages: int = 1,
    wait_time_seconds: int = 1) -> list[Message]:
    """
    Fetches messages from an SQS queue.

    Args:
    queue_url : str
        The URL of the SQS queue.
    max_messages : int, optional
        The maximum number of messages to retrieve (default is 1).
    wait_time_seconds : int, optional
        The duration (in seconds) for which the call will wait for a message to arrive 
        in the queue before returning (default is 1).

    Returns:
    list[Message]
        A list of Message objects representing the fetched SQS messages.
    """
    sqs = boto3.client("sqs")
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
    except ClientError as err:
        logger.error(err)
        return []
    if "Messages" not in response:
        return []
    return [Message(m["ReceiptHandle"], m["Body"]) for m in response["Messages"]]


def delete_message(queue_url: str, receipt_handle: str):
    """
    Deletes a specific message from an SQS queue.

    Args:
    queue_url : str
        The URL of the SQS queue.
    receipt_handle : str
        The receipt handle of the message to delete.
    """
    sqs = boto3.client("sqs")
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
