from common import read_params
import argparse
import sentencepiece as spm


def create_tokenizer(config_path):
    config = read_params(config_path)
    pretraining_txt = config["data"]["pretraining_txt"]
    vocab_size = config["hyperparameters"]["vocab_size"]
    vocab_dir = config["data"]["vocab_dir"]

    spm.SentencePieceTrainer.train(
        input=pretraining_txt, model_prefix=vocab_dir, vocab_size=vocab_size
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    create_tokenizer(config_path=parsed_args.config)
