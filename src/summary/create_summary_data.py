from src.common import read_params
import pandas as pd
import argparse
import tiktoken
# import re
import tqdm


def clean_text_pipeline(text):
    '''# Less aggressive cleaning - preserve more semantic content
    text = text.replace("-lrb-", "(").replace("-rrb-", ")")
    text = text.replace("-lsb-", "[").replace("-rsb-", "]")
    
    # Keep parenthetical content but remove citations
    text = re.sub(r"\(\d+\)", "", text)  # Remove citation numbers
    
    # Replace URLs and mentions with placeholder tokens instead of removing
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[MENTION]", text)
    
    # Remove non-ASCII but preserve common unicode punctuation
    text = re.sub(r"[^\x00-\x7F\u2013\u2014\u2018\u2019\u201C\u201D]", "", text)
    
    # Keep more punctuation that might be semantically important
    text = re.sub(r"[^\w\s\-'\".,!?:;()\[\]]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()'''
    
    return text


def create_summary_data(config_path):
    config = read_params(config_path)
    summarization_full_txt = config["data"]["summarization_full_txt"]
    summarization_summary_txt = config["data"]["summarization_summary_txt"]
    summarization_filtered_parquet = config["data"]["summarization_filtered_parquet"]
    block_size = config["hyperparameters"]["block_size"]
    summary_size = config["data"]["summary_size"]

    # Load the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(summarization_full_txt, "r") as file1:
        full_length = file1.readlines()
    with open(summarization_summary_txt, "r") as file1:
        summary = file1.readlines()

    # Set the target limit
    data = []

    with tqdm.tqdm(total=summary_size, desc="Creating summary training data") as pbar:

        for line in range(len(full_length)):

            X = f"Summarize the following article.\n\n### Article:\n{full_length[line][:-1]}\n\n### Summary:\n{summary[line][:-1]}<|endoftext|>"
            X = tokenizer.encode(X, allowed_special="all")
            
            if len(X) <= block_size:

                # X
                X += [27156] * (block_size - len(X))


                # y
                y = X.copy()
                y = X[1:] + [27156]

                # attention mask
                att_mask = [1 if id != 27156 else 0 for id in X]

                data.append([X, y, att_mask])

                pbar.update(1)

            if len(data) >= summary_size:
                break

    df = pd.DataFrame(data, columns=["article", "summary", "attention_mask"])

    df.to_parquet(summarization_filtered_parquet, engine="pyarrow")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_summary_data(config_path=parsed_args.config)
