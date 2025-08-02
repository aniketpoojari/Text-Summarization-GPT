import os
import json

import torch
from torch.nn import functional as F

import deepspeed
import deepspeed.comm as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam

from model import GPT
from loss import estimate_loss_pretraining
from lr_scheduler import CustomCosineWarmupScheduler
from dataloader import DeepSpeedDataLoaderPretraining

import mlflow

def setup_distributed():
    """
    Initialize distributed training using DeepSpeed's communication module.
    On SageMaker, environment variables (SM_HOSTS, SM_CURRENT_HOST) determine the cluster configuration.
    """
    try:
        # Parse SageMaker environment variables
        sm_hosts = json.loads(os.environ.get('SM_HOSTS'))
        sm_current_host = os.environ.get('SM_CURRENT_HOST')
        world_size = len(sm_hosts)
        rank = sm_hosts.index(sm_current_host)
        local_rank = 0  # Typically one GPU per instance in this setup

        # Set the usual env variables (DeepSpeed still honors these under the hood)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        master_addr = sm_hosts[0]
        master_port = '29500'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # Initialize distributed via DeepSpeed
        dist.init_distributed()

        # Set device
        torch.cuda.set_device(local_rank)

        return rank, world_size
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")


def training():
 
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Clears unused memory
    torch.cuda.empty_cache()

    rank, world_size = setup_distributed()
    device = "cuda"
    
    ###### Hyperparameters  ######
    vocab_size = int(os.getenv("VOCAB_SIZE", "50257"))
    n_embd = int(os.getenv("N_EMBD", "768"))
    block_size = int(os.getenv("BLOCK_SIZE", "128"))
    n_head = int(os.getenv("N_HEAD", "12"))
    n_layer = int(os.getenv("N_LAYER", "12"))
    dropout = float(os.getenv("DROPOUT", "0.2"))
    epochs = int(os.getenv("EPOCHS", "3"))
    no_of_mini_batches = int(os.getenv("MAX_ITERS", "2500"))
    batch_size = int(os.getenv("BATCH_SIZE", "256"))
    mini_batch_size = int(os.getenv("MINI_BATCH_SIZE", "16"))
    learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
    eval_iters = int(os.getenv("EVAL_ITERS", "50"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "500"))

    ###### MLFlow  ######
    if rank == 0:

        # Get environment variables
        experiment_name = os.getenv("EXPERIMENT_NAME")
        run_name = os.getenv("RUN_NAME")
        registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
        server_uri = os.getenv("SERVER_URI")
        s3_mlruns_bucket = os.getenv("S3_MLRUNS_BUCKET")

        # Initialize MLFlow
        mlflow.set_tracking_uri(server_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, s3_mlruns_bucket)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

        # Log hyperparameters
        mlflow.log_params(
            {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "block_size": block_size,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "epochs": epochs,
                "no_of_mini_batches": no_of_mini_batches,
                "batch_size": batch_size,
                "mini_batch_size": mini_batch_size,
                "learning_rate": learning_rate,
                "eval_interval": eval_interval,
                "eval_iters": eval_iters,
                "device": device,
            }
        )
    
    ###### Data  ######
    pretraining_train_path = "/opt/ml/input/data/train/pretraining_train.pkl"
    pretraining_val_path = "/opt/ml/input/data/train/pretraining_val.pkl"

    ###### Model  ######
    model = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device).to(
        device
    )

    if rank == 0:
        print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")


    # Create DeepSpeed optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = DeepSpeedCPUAdam(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Create DeepSpeed scheduler
    scheduler = CustomCosineWarmupScheduler(
        optimizer,
        max_iters=no_of_mini_batches,
        warmup_iters=int(no_of_mini_batches * 0.1),
        min_lr_ratio=0.1
    )

    ds_config = {
        "train_batch_size": batch_size * world_size,
        "gradient_accumulation_steps": batch_size // mini_batch_size,
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 4,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
        },
        "gradient_clipping": 0.5
    }


    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=scheduler
    )

    ###### Set Precision  ######
    torch.set_float32_matmul_precision("medium")
    
    # Create Train Dataloader Once
    train_dataloader = DeepSpeedDataLoaderPretraining(
            batch_size=batch_size,
            seq_length=block_size,
            data_path=pretraining_train_path,
            device=device,
            rank=rank,  
            world=world_size
        )


    ###### Epochs  ######
    for epoch in range(epochs):
        
        # Set epoch for the sampler
        train_dataloader.sampler.set_epoch(epoch)
        
        # Reset train dataloader to start at the start of each epoch
        train_dataloader.reset()

        if rank == 0:
            print(f"###### Epoch {epoch + 1}/{epochs} ######")

        # Mini Batches
        for mini_batch in range(no_of_mini_batches):
        
            ###### Eval  ######
            if rank == 0 and (mini_batch % eval_interval == 0 or mini_batch == no_of_mini_batches - 1):
                losses = estimate_loss_pretraining(
                    model,
                    pretraining_train_path,
                    pretraining_val_path,
                    eval_iters,
                    block_size,
                    batch_size,
                    mini_batch_size,
                    device,
                    rank,
                    world_size
                )

                print(
                    f"step {mini_batch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                mlflow.log_metric("Train Loss", losses["train"], step=mini_batch)
                mlflow.log_metric("Val Loss", losses["val"], step=mini_batch)

            ###### Train  ######

            model.train()
            
            xb, yb = train_dataloader.next_batch()

            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(xb)

                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = yb.view(B * T)
                mini_batch_loss = F.cross_entropy(logits, targets)

            # backward pass
            model.backward(mini_batch_loss)
            model.step()
            lr_scheduler.step()


    # save the model
    if rank == 0:
        print("\n\n\nTraining complete")
        
        print("\nCreating GPT Model")
        model_gpt = GPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout, "cpu")
        model_gpt.load_state_dict(model.module.state_dict())
        
        print("\nSaving models to MLflow")
        mlflow.pytorch.log_model(model_gpt, "GPT", registered_model_name=f"{registered_model_name}")
        
        print("\nModels successfully saved to MLflow!")
        mlflow.end_run()

if __name__ == "__main__":
    training()
