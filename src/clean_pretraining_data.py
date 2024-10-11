from common import read_params
import argparse
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


def create_pretraining_data(config_path):
    config = read_params(config_path)
    summarization_full_txt = config["data"]["summarization_full_txt"]
    pretraining_txt = config["data"]["pretraining_txt"]

    text = ""
    with open(summarization_full_txt, "r", encoding="utf-8") as f:
        total_lines = sum(
            1 for _ in open(summarization_full_txt, "r", encoding="utf-8")
        )
        for line in tqdm.tqdm(f, total=total_lines, desc="Reading file"):
            text += clean_text_pipeline(line) + " "

    with open(pretraining_txt, "w") as file:
        file.write(text)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_pretraining_data(config_path=parsed_args.config)
