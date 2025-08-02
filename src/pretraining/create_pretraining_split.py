from src.common import read_params
import argparse
import pickle


def create_split(config_path):
    config = read_params(config_path)
    pretraining_pkl = config["data"]["pretraining_pkl"]
    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]
    pretraining_split = config["data"]["pretraining_split"]

    # LOAD PKL FILE
    with open(pretraining_pkl, "rb") as file:
        encoding = pickle.load(file)

    # SPLIT SIZE
    n = int(pretraining_split * len(encoding))

    # SPLITS
    train_data = encoding[:n]
    val_data = encoding[n:]

    with open(pretraining_train, "wb") as file:
        pickle.dump(train_data, file)
    with open(pretraining_val, "wb") as file:
        pickle.dump(val_data, file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_split(config_path=parsed_args.config)
