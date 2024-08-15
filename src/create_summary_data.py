from common import read_params
import pandas as pd
import argparse
import sentencepiece as spm


def create_summary_data(config_path):
    config = read_params(config_path)
    summarization_parquet = config["data"]["summarization_parquet"]
    summarization_filtered_parquet = config["data"]["summarization_filtered_parquet"]
    block_size = config["hyperparameters"]["block_size"]
    vocab_dir = config["data"]["vocab_dir"] + ".model"

    train = pd.read_parquet(summarization_parquet, columns=["article", "summary"])
    tokenizer = spm.SentencePieceProcessor(model_file=vocab_dir)

    indices = []
    for i in range(train.shape[0]):
        tokens = tokenizer.encode(train.iloc[i]["article"], out_type=int)
        if len(tokens) <= block_size:
            indices.append(i)
            train.iloc[i, 0] = tokens + [1] * (block_size - len(tokens))
            tokens = tokenizer.encode(train.iloc[i]["summary"], out_type=int)
            train.iloc[i, 1] = tokens + [1] * (block_size - len(tokens))

    train = train.iloc[indices]

    train = train.reset_index(drop=True)

    train.to_parquet(summarization_filtered_parquet, engine="pyarrow")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_summary_data(config_path=parsed_args.config)
