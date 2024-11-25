from common import read_params
import argparse
import pickle
from transformers import GPT2Tokenizer


def create_split(config_path):
    config = read_params(config_path)
    pretraining_txt = config["data"]["pretraining_txt"]
    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]

    # LOAD THE COMPLETE FILE
    with open(pretraining_txt, "r") as file:
        text = file.read()

    # Load the GPT-2 tokenizer and add special tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<PAD>", "sep_token": "<SEP>"})

    text = tokenizer.encode(text)

    # SPLIT SIZE
    n = int(0.7 * len(text))  # first 90% will be train, rest val

    # SPLITS
    train_data = text[:n]
    val_data = text[n:]

    with open(pretraining_train, "wb") as file:
        pickle.dump(train_data, file)
    with open(pretraining_val, "wb") as file:
        pickle.dump(val_data, file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_split(config_path=parsed_args.config)
