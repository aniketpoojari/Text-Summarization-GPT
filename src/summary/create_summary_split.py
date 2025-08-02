from src.common import read_params
import argparse
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import torch

def create_split(config_path):
    config = read_params(config_path)
    summarization_filtered_parquet = config["data"]["summarization_filtered_parquet"]
    summary_train_dir = config["data"]["summary_train"]
    summary_val_dir = config["data"]["summary_val"]

    # LOAD THE COMPLETE FILE
    summarization_filtered_parquet = pd.read_parquet(summarization_filtered_parquet)

    # SHUFFLE
    summarization_filtered_parquet = shuffle(summarization_filtered_parquet, random_state=42)

    # SPLIT SIZE
    n = int(0.8 * len(summarization_filtered_parquet))

    # SPLITS
    train = summarization_filtered_parquet.iloc[:n, :]
    val = summarization_filtered_parquet.iloc[n:, :]

    # SAVE AS PARQUET
    train = torch.tensor(np.array(train.values.tolist()))
    val = torch.tensor(np.array(val.values.tolist()))

    torch.save(train, summary_train_dir)
    torch.save(val, summary_val_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_split(config_path=parsed_args.config)
