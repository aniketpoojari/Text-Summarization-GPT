from src.common import read_params
import argparse
import boto3
from datetime import datetime


def push_pretraining_data_to_s3(config_path):
    config = read_params(config_path)
    pretraining_s3 = config["data"]["pretraining_s3"]
    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]

    s3Resource = boto3.resource('s3')

    s3Resource.meta.client.upload_file(pretraining_train, pretraining_s3, pretraining_train.split('/')[-1])
    s3Resource.meta.client.upload_file(pretraining_val, pretraining_s3, pretraining_val.split('/')[-1])

    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("push_pretraining_data_to_s3_completion.txt", "w") as file:
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Write the formatted date and time to the file
        file.write("Pretraining data pushed to S3 at " + formatted_datetime)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    push_pretraining_data_to_s3(config_path=parsed_args.config)
