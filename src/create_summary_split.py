from common import read_params
import argparse
import pickle
import pandas as pd
from sklearn.utils import shuffle


def create_split(config_path):
    config = read_params(config_path)
    summarization_filtered_parquet = config["data"]["summarization_filtered_parquet"]
    train_full = config["data"]["train_full"]
    train_summary = config["data"]["train_summary"]
    val_full = config["data"]["val_full"]
    val_summary = config["data"]["val_summary"]

    # LOAD THE COMPLETE FILE
    summarization_filtered_parquet = pd.read_parquet(summarization_filtered_parquet)

    # SHUFFLE
    summarization_filtered_parquet = shuffle(
        summarization_filtered_parquet, random_state=42
    )

    # SPLIT SIZE
    n = int(0.8 * len(summarization_filtered_parquet))

    # SPLITS
    train_X = summarization_filtered_parquet.iloc[:n, 0].tolist()
    train_y = summarization_filtered_parquet.iloc[:n, 1].tolist()
    val_X = summarization_filtered_parquet.iloc[n:, 0].tolist()
    val_y = summarization_filtered_parquet.iloc[n:, 1].tolist()

    with open(train_full, "wb") as file:
        pickle.dump(train_X, file)
    with open(train_summary, "wb") as file:
        pickle.dump(train_y, file)
    with open(val_full, "wb") as file:
        pickle.dump(val_X, file)
    with open(val_summary, "wb") as file:
        pickle.dump(val_y, file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_split(config_path=parsed_args.config)
