from common import read_params
import argparse
import sentencepiece as spm


def create_split(config_path):
    config = read_params(config_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_split(config_path=parsed_args.config)
