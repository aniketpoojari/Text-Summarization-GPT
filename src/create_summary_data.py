from common import read_params
import pandas as pd
import argparse
from transformers import GPT2Tokenizer
import re
import tqdm


def clean_text_pipeline(text):
    # Replace custom markers with actual characters
    text = text.replace("-lrb-", "(")
    text = text.replace("-rrb-", ")")
    text = text.replace("-lsb-", "[")  # Assuming -lsb- represents left square bracket
    text = text.replace("-rsb-", "]")  # Assuming -rsb- represents right square bracket

    # Define regex pattern to match text within parentheses
    text = re.sub(r"\([^\)]*\)", "", text)

    # Replace URLs and mentions
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Keep letters, numbers, and whitespace
    text = re.sub(r"[^A-Za-z0-9\s\-'\"]", "", text)

    return text


def create_summary_data(config_path):
    config = read_params(config_path)
    summarization_full_txt = config["data"]["summarization_full_txt"]
    summarization_summary_txt = config["data"]["summarization_summary_txt"]
    summarization_filtered_parquet = config["data"]["summarization_filtered_parquet"]
    block_size = config["hyperparameters"]["block_size"]

    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add special tokens
    special_tokens_dict = {"pad_token": "<PAD>", "sep_token": "<SEP>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    with open(summarization_full_txt, "r") as file1:
        full_length = file1.readlines()
    with open(summarization_summary_txt, "r") as file1:
        summary = file1.readlines()

    train = pd.DataFrame(columns=["article", "summary"])

    for line in tqdm.tqdm(range(len(full_length)), desc="Reading file"):
        full = tokenizer.encode(clean_text_pipeline(full_length[line]))
        summ = tokenizer.encode(clean_text_pipeline(summary[line]))
        l = len(full) + 1 + len(summ)
        if l <= block_size:
            # X
            X = full + [tokenizer.sep_token_id] + summ
            X += [tokenizer.pad_token_id] * (block_size - len(X))

            # y
            y = [-1] * len(full) + [tokenizer.sep_token_id] + summ
            y += [tokenizer.pad_token_id] * (block_size - len(y))

            train.loc[len(train.index)] = [X, y]
        if train.shape[0] == 50000:
            break

    train.to_parquet(summarization_filtered_parquet, engine="pyarrow")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_summary_data(config_path=parsed_args.config)
