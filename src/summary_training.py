from common import read_params
import torch
import argparse
from model import estimate_loss_summary, GPT
from data_loader import DataLoaderSummary
import pickle
from torch.nn import functional as F
import mlflow
from datetime import datetime
import math
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from rouge_score import rouge_scorer
from transformers import GPT2Tokenizer


def rouge_loss(predictions, targets, tokenizer):

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_l_scores = []
    for prediction, target in zip(predictions, targets):
        # Just remove the full text
        mask = target != -1
        prediction = prediction[mask][:-1]
        target = target[mask][1:]

        # Decode
        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        target = tokenizer.decode(target, skip_special_tokens=True)

        # Score
        score = scorer.score(target, prediction)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    # Averaging across all samples
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return 1 - avg_rouge_l


def summary_training(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_params(config_path)

    ###### Hyperparameters  ######
    pretrained_model = config["log_pretrained_model"]["model_dir"]
    learning_rate = float(config["summary_hyperparameters"]["learning_rate"])
    epochs = config["summary_hyperparameters"]["epochs"]
    max_iters = config["summary_hyperparameters"]["max_iters"]
    batch_size = config["summary_hyperparameters"]["batch_size"]
    mini_batch_size = config["summary_hyperparameters"]["mini_batch_size"]
    steps = batch_size // mini_batch_size
    eval_interval = config["summary_hyperparameters"]["eval_interval"]
    eval_iters = config["summary_hyperparameters"]["eval_iters"]

    ###### Data  ######
    train_full = config["data"]["train_full"]
    train_summary = config["data"]["train_summary"]
    val_full = config["data"]["val_full"]
    val_summary = config["data"]["val_summary"]

    ###### Load Data  ######
    with open(train_full, "rb") as file:
        train_X = torch.tensor(np.array(pickle.load(file)), dtype=torch.long)
    with open(train_summary, "rb") as file:
        train_y = torch.tensor(np.array(pickle.load(file)), dtype=torch.long)
    with open(val_full, "rb") as file:
        val_X = torch.tensor(np.array(pickle.load(file)), dtype=torch.long)
    with open(val_summary, "rb") as file:
        val_y = torch.tensor(np.array(pickle.load(file)), dtype=torch.long)

    ###### MLFlow  ######
    experiment_name = config["mlflow_summary"]["experiment_name"]
    run_name = config["mlflow_summary"]["run_name"]
    registered_model_name = config["mlflow_summary"]["registered_model_name"]
    server_uri = config["mlflow_summary"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    ###### Set Precision  ######
    torch.set_float32_matmul_precision("medium")

    ###### Load Model  ######
    model = torch.load(pretrained_model, map_location=device, weights_only=False)

    ###### Learning Rate Scheduler  ######
    max_lr = learning_rate
    min_lr = 1e-6
    max_steps = max_iters
    warmup_steps = max_iters * 0.1

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    ###### Optimizer  ######
    optimizer = model.configure_optimizers(
        weight_decay=0.01, learning_rate=learning_rate, device_type=device
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ###### Scheduler  ######
    # sheduler = CosineAnnealingLR(optimizer, max_iters, eta_min=3e-5)

    ###### Dataloaders  ######
    train_dataloader = DataLoaderSummary(mini_batch_size, train_X, train_y, device)
    val_train_dataloader = DataLoaderSummary(mini_batch_size, train_X, train_y, device)
    val_val_dataloader = DataLoaderSummary(mini_batch_size, val_X, val_y, device)

    ###### Tokenizer  ######
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {"pad_token": "<PAD>", "sep_token": "<SEP>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    ###### Training  ######
    with mlflow.start_run(run_name=run_name) as mlflow_run:

        ###### Epochs  ######
        for epoch in range(epochs):

            print("##### Epoch - ", epoch + 1)

            # set dataloader to the start
            train_dataloader.reset()

            ##### Iterations  ######
            for iter in range(max_iters):

                #### Eval  ######
                if iter % eval_interval == 0 or iter == max_iters - 1:
                    losses = estimate_loss_summary(
                        model,
                        batch_size,
                        mini_batch_size,
                        device,
                        eval_iters,
                        val_train_dataloader,
                        val_val_dataloader,
                        tokenizer,
                    )
                    print(
                        f"step {iter}: train loss {losses['train']:.4f}, train_rouge {losses['train_rouge']:.4f}, val loss {losses['val']:.4f}, val_rouge {losses['val_rouge']:.4f}"
                    )
                    mlflow.log_metric("Train Loss", losses["train"], step=iter)
                    mlflow.log_metric("Val Loss", losses["val"], step=iter)
                    mlflow.log_metric("Train Rouge", losses["train_rouge"], step=iter)
                    mlflow.log_metric("Val Rouge", losses["val_rouge"], step=iter)

                ##### Train  ######

                # set model to train mode
                model.train()

                # zero the gradients
                optimizer.zero_grad()

                #### Mini Batch  ######
                for _ in range(steps):

                    # sample a batch of data
                    xb, yb = train_dataloader.next_batch()

                    # cast to bfloat16
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):

                        # get prediction
                        logits = model(xb)

                        # get rouge loss
                        rouge = rouge_loss(logits.argmax(dim=-1), yb, tokenizer)

                        # reshape
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = yb.view(B * T)

                        # get proper part of output and target
                        valid_mask = targets != -1
                        logits = logits[valid_mask][:-1]
                        targets = targets[valid_mask][1:]
                        loss = F.cross_entropy(logits, targets) + rouge

                    # average loss
                    loss = loss / steps

                    # backprop
                    loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # learning rate
                lr = get_lr(iter)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # optimizer step
                optimizer.step()

                # sheduler.step()

        # save the model
        mlflow.pytorch.log_model(
            model,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )

    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("training_summary_completion.txt", "w") as file:
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Write the formatted date and time to the file
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    summary_training(config_path=parsed_args.config)
