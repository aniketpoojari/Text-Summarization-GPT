from common import read_params
from model import BigramLanguageModel
import torch
import argparse
from model import estimate_loss
from data_loader import get_batch
import pickle
from torch.nn import functional as F
import mlflow
from datetime import datetime


def training(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_params(config_path)
    vocab_size = config["hyperparameters"]["vocab_size"]
    n_embd = config["hyperparameters"]["n_embd"]
    block_size = config["hyperparameters"]["block_size"]
    n_head = config["hyperparameters"]["n_head"]
    n_layer = config["hyperparameters"]["n_layer"]
    dropout = config["hyperparameters"]["dropout"]
    batch_size = config["hyperparameters"]["batch_size"]
    learning_rate = float(config["hyperparameters"]["learning_rate"])
    max_iters = config["hyperparameters"]["max_iters"]
    eval_iters = config["hyperparameters"]["eval_iters"]
    eval_interval = config["hyperparameters"]["eval_interval"]

    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]

    with open(pretraining_train, "rb") as file:
        train = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(pretraining_val, "rb") as file:
        val = torch.tensor(pickle.load(file), dtype=torch.long)

    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]
    server_uri = config["mlflow"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    model = BigramLanguageModel(
        vocab_size, n_embd, block_size, n_head, n_layer, dropout, device
    ).to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    with mlflow.start_run(run_name=run_name) as mlflow_run:

        mlflow.log_params(
            {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "block_size": block_size,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_iters": max_iters,
            }
        )

        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(
                    model,
                    "pretraining",
                    eval_iters,
                    block_size,
                    batch_size,
                    device,
                    train,
                    val,
                )

                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                mlflow.log_metric("Train Loss", losses["train"], step=iter)
                mlflow.log_metric("Val Loss", losses["val"], step=iter)

            # sample a batch of data
            xb, yb = get_batch(
                "pretraining",
                "train",
                block_size,
                batch_size,
                device,
                pre_train=train,
                pre_val=val,
            )

            # get prediction
            logits = model(xb)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = yb.view(B * T)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        mlflow.pytorch.log_model(
            model,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )

    with open("training_completion.txt", "w") as file:
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
