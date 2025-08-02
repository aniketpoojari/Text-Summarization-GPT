from src.common import read_params
import argparse
# import re
import tqdm
import pickle
import tiktoken

def create_pretraining_data(config_path):
    # Read config
    config = read_params(config_path)
    summarization_full_txt = config["data"]["summarization_full_txt"]
    pretraining_pkl = config["data"]["pretraining_pkl"]
    pretraining_size = config["data"]["pretraining_size"]

    # Load the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Clean the data
    text = []
    with open(summarization_full_txt, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        print("Total lines in the file:", total_lines)

        if pretraining_size > total_lines:
            print("Count is greater than total lines in the file.")
            return

        with tqdm.tqdm(total=pretraining_size, desc="Creating summary training data") as pbar:
            i = 0
            for line in f:
                # text.extend(tokenizer.encode(clean_text_pipeline(line)))
                text.extend(tokenizer.encode(line, allowed_special="all"))
                pbar.update(1)
                i += 1
                if i == pretraining_size:
                    break

    # Write to pinkle file
    with open(pretraining_pkl, "wb") as file:
        pickle.dump(text, file)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_pretraining_data(config_path=parsed_args.config)
