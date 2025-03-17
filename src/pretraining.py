from common import read_params
import argparse
import torch
from torch.nn import functional as F
from data_loader import DataLoaderPretraining
from model import GPT, estimate_loss_pretraining
import mlflow
import pickle
import math
from datetime import datetime


def training(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_params(config_path)

    ###### Hyperparameters  ######
    vocab_size = config["hyperparameters"]["vocab_size"]
    n_embd = config["hyperparameters"]["n_embd"]
    block_size = config["hyperparameters"]["block_size"]
    n_head = config["hyperparameters"]["n_head"]
    n_layer = config["hyperparameters"]["n_layer"]
    dropout = config["hyperparameters"]["dropout"]
    epochs = config["hyperparameters"]["epochs"]
    max_iters = config["hyperparameters"]["max_iters"]
    batch_size = config["hyperparameters"]["batch_size"]
    mini_batch_size = config["hyperparameters"]["mini_batch_size"]
    steps = batch_size // mini_batch_size
    learning_rate = float(config["hyperparameters"]["learning_rate"])
    eval_iters = config["hyperparameters"]["eval_iters"]
    eval_interval = config["hyperparameters"]["eval_interval"]

    ###### Data  ######
    pretraining_train = config["data"]["pretraining_train"]
    pretraining_val = config["data"]["pretraining_val"]

    ###### Load data  ######
    with open(pretraining_train, "rb") as file:
        train = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(pretraining_val, "rb") as file:
        val = torch.tensor(pickle.load(file), dtype=torch.long)

    ###### MLFlow  ######
    experiment_name = config["mlflow_pretraining"]["experiment_name"]
    run_name = config["mlflow_pretraining"]["run_name"]
    registered_model_name = config["mlflow_pretraining"]["registered_model_name"]
    server_uri = config["mlflow_pretraining"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    ###### Set Precision  ######
    torch.set_float32_matmul_precision("medium")

    ###### Model  ######
    model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device).to(
        device
    )
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    ###### Learning Rate Scheduler  ######
    max_lr = learning_rate
    min_lr = max_lr * 0.1
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
        weight_decay=0.1, learning_rate=learning_rate, device_type=device
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    ###### Scheduler  ######
    # sheduler = CosineAnnealingLR(optimizer, max_iters, eta_min=1e-5)

    ###### Dataloader  ######
    train_dataloader = DataLoaderPretraining(mini_batch_size, block_size, train, device)
    val_train_dataloader = DataLoaderPretraining(
        mini_batch_size, block_size, train, device
    )
    val_val_dataloader = DataLoaderPretraining(mini_batch_size, block_size, val, device)

    ###### Training  ######
    with mlflow.start_run(run_name=run_name) as mlflow_run:

        # log hyperparameters
        mlflow.log_params(
            {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "block_size": block_size,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "epochs": epochs,
                "max_iters": max_iters,
                "batch_size": batch_size,
                "mini_batch_size": mini_batch_size,
                "learning_rate": learning_rate,
                "eval_interval": eval_interval,
                "eval_iters": eval_iters,
                "device": device,
            }
        )

        ###### Epochs  ######
        for epoch in range(epochs):

            print("##### Epoch - ", epoch + 1)

            # set dataloader to the start
            train_dataloader.reset()

            ##### Iterations  ######
            for iter in range(max_iters):

                ###### Eval  ######
                if iter % eval_interval == 0 or iter == max_iters - 1:
                    losses = estimate_loss_pretraining(
                        model,
                        128,
                        mini_batch_size,
                        device,
                        eval_iters,
                        val_train_dataloader,
                        val_val_dataloader,
                    )

                    print(
                        f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )
                    mlflow.log_metric("Train Loss", losses["train"], step=iter)
                    mlflow.log_metric("Val Loss", losses["val"], step=iter)

                ###### Train  ######

                # set model to training
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

                        # loss calculation
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = yb.view(B * T)
                        loss = F.cross_entropy(logits, targets)

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
    with open("training_completion.txt", "w") as file:
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
    training(config_path=parsed_args.config)
