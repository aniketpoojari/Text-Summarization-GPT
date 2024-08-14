from common import read_params
import pandas as pd
import argparse


def create_pretraining_data(config_path):
    config = read_params(config_path)
    pretraining_parquet = config["data"]["pretraining_parquet"]
    pretraining_txt = config["data"]["pretraining_txt"]

    file = pd.read_parquet(pretraining_parquet)

    with open(pretraining_txt, "a", encoding="utf-8") as f:
        for i in range(file.shape[0]):
            f.write(file.iloc[i].text + "\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_pretraining_data(config_path=parsed_args.config)
