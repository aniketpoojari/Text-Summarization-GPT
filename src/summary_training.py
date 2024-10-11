from common import read_params
import torch
import argparse
from model import estimate_loss
from data_loader import get_batch
import pickle
from torch.nn import functional as F
import mlflow
from datetime import datetime


def summary_training(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_params(config_path)
    pretrained_model = config["log_pretrained_model"]["model_dir"]
    learning_rate = float(config["summary_hyperparameters"]["learning_rate"])
    max_iters = config["summary_hyperparameters"]["max_iters"]
    eval_interval = config["hyperparameters"]["eval_interval"]
    block_size = config["hyperparameters"]["block_size"]
    batch_size = config["summary_hyperparameters"]["batch_size"]
    eval_iters = config["hyperparameters"]["eval_iters"]

    train_full = config["data"]["train_full"]
    train_summary = config["data"]["train_summary"]
    val_full = config["data"]["val_full"]
    val_summary = config["data"]["val_summary"]

    with open(train_full, "rb") as file:
        train_X = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(train_summary, "rb") as file:
        train_y = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(val_full, "rb") as file:
        val_X = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(val_summary, "rb") as file:
        val_y = torch.tensor(pickle.load(file), dtype=torch.long)

    experiment_name = config["mlflow_summary"]["experiment_name"]
    run_name = config["mlflow_summary"]["run_name"]
    registered_model_name = config["mlflow_summary"]["registered_model_name"]
    server_uri = config["mlflow_summary"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    model = torch.load(pretrained_model).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        for iter in range(max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(
                    model,
                    "summary",
                    eval_iters,
                    block_size,
                    batch_size,
                    device,
                    train_full=train_X,
                    train_summ=train_y,
                    val_full=val_X,
                    val_summ=val_y,
                )
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, train_rouge {losses['train_rouge']:.4f}, val loss {losses['val']:.4f}, val_rouge {losses['val_rouge']:.4f}"
                )
                mlflow.log_metric("Train Loss", losses["train"], step=iter)
                mlflow.log_metric("Val Loss", losses["val"], step=iter)

            # sample a batch of data
            xb, yb = get_batch(
                "summary",
                "train",
                block_size,
                batch_size,
                device,
                train_full=train_X,
                train_summ=train_y,
                val_full=val_X,
                val_summ=val_y,
            )

            # get prediction
            logits = model(xb)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = yb.view(B * T)

            valid_mask = targets != -1
            targets = targets[valid_mask][1:]
            logits = logits[valid_mask][:-1]

            loss = F.cross_entropy(logits, targets, ignore_index=50257)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        mlflow.pytorch.log_model(
            model,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )

    with open("training_summary_completion.txt", "w") as file:
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    summary_training(config_path=parsed_args.config)
