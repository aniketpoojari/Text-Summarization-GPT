from common import read_params
import argparse
import sentencepiece as spm
import torch
import pickle


def create_split(config_path):
    config = read_params(config_path)
    pretraining_txt = config["data"]["pretraining_txt"]
    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]
    vocab_dir = config["data"]["vocab_dir"] + ".model"

    # LOAD THE COMPLETE FILE
    with open(pretraining_txt, "r", encoding="utf-8") as f:
        text = f.read()

    # LOAD THE TOKENIZER
    tokenizer = spm.SentencePieceProcessor(model_file=vocab_dir)

    # ENCODE ENTIRE DATA
    text = tokenizer.encode(text, out_type=int)

    # TRAIN TEST SPLIT
    # text = torch.tensor(text, dtype=torch.long)

    # SPLIT SIZE
    n = int(0.9 * len(text))  # first 90% will be train, rest val

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
