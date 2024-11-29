import torch
from torch.nn import functional as F
from dataloader import DataLoaderPretraining
from model import GPT, estimate_loss_pretraining
import pickle
import math
import os
import argparse


def training():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###### Hyperparameters  ######
    vocab_size = int(os.environ.get("vocab_size", 50257))  # Default if not set
    n_embd = int(os.environ.get("n_embd", 768))  # Default if not set
    block_size = int(os.environ.get("block_size", 128))  # Default if not set
    n_head = int(os.environ.get("n_head", 12))  # Default if not set
    n_layer = int(os.environ.get("n_layer", 12))  # Default if not set
    dropout = float(os.environ.get("dropout", 0.1))  # Default if not set
    epochs = int(os.environ.get("epochs", 1))  # Default if not set
    max_iters = int(os.environ.get("max_iters", 100))  # Default if not set
    batch_size = int(os.environ.get("batch_size", 32))  # Default if not set
    mini_batch_size = int(os.environ.get("mini_batch_size", 16))  # Default if not set
    learning_rate = float(os.environ.get("learning_rate", 2e-4))  # Default if not set
    eval_iters = int(os.environ.get("eval_iters", 50))  # Default if not set
    eval_interval = int(os.environ.get("eval_interval", 100))  # Default if not set
    steps = batch_size // mini_batch_size

    # Define the paths based on SageMaker's input channel structure
    train_data_path = "/opt/ml/input/data/train/pretraining_train.pkl"
    val_data_path = "/opt/ml/input/data/validation/pretraining_val.pkl"

    # Load data locally after download
    with open(train_data_path, "rb") as file:
        train = torch.tensor(pickle.load(file), dtype=torch.long)
    with open(val_data_path, "rb") as file:
        val = torch.tensor(pickle.load(file), dtype=torch.long)

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
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    ###### Optimizer  ######
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=learning_rate, device_type=device
    )

    ###### Dataloader  ######
    train_dataloader = DataLoaderPretraining(mini_batch_size, block_size, train, device)
    val_train_dataloader = DataLoaderPretraining(
        mini_batch_size, block_size, train, device
    )
    val_val_dataloader = DataLoaderPretraining(mini_batch_size, block_size, val, device)

    ###### Training  ######
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

            ###### Train  ######
            model.train()

            # zero the gradients
            optimizer.zero_grad()

            # Mini Batch
            for _ in range(steps):
                xb, yb = train_dataloader.next_batch()

                # cast to bfloat16
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(xb)
                    B, T, C = logits.shape
                    logits = logits.view(B * T, C)
                    targets = yb.view(B * T)
                    loss = F.cross_entropy(logits, targets)

                loss = loss / steps
                loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # learning rate
            lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

    ###### Save Model  ######
    output_dir = "/opt/ml/model/"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model, os.path.join(output_dir, "model.pth"))


if __name__ == "__main__":
    training()
