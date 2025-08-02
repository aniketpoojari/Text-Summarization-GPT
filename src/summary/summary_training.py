from src.common import read_params
import mlflow
import inspect
import argparse
from datetime import datetime
import tiktoken
import torch
from torch.nn import functional as F

from src.summary.loss import estimate_loss_summary, cross_entropy_loss, rouge_score
from src.summary.dataloader import DataLoaderSummary
from src.pretraining.code.lr_scheduler import CustomCosineWarmupScheduler

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
    train_dir = config["data"]["summary_train"]
    val_dir = config["data"]["summary_val"]

    ###### Load Data  ######
    train = torch.load(train_dir)
    val = torch.load(val_dir)

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


    ###### Optimizer  ######
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )

    ###### Scheduler  ######
    scheduler = CustomCosineWarmupScheduler(
        optimizer,
        max_iters=max_iters,
        warmup_iters=int(max_iters * 0.1),
        min_lr_ratio=0.1
    )
 
    ###### Dataloaders  ######
    train_dataloader = DataLoaderSummary(mini_batch_size, train, device)
    val_train_dataloader = DataLoaderSummary(mini_batch_size, train, device)
    val_val_dataloader = DataLoaderSummary(mini_batch_size, val, device)

    ###### Tokenizer  ######
    tokenizer = tiktoken.get_encoding("gpt2")
    
    ###### Training  ######
    with mlflow.start_run(run_name=run_name) as mlflow_run:

        ###### Epochs  ######
        for epoch in range(epochs):

            print("##### Epoch - ", epoch + 1)

            # set dataloader to the start
            train_dataloader.reset()

            ##### Iterations  ######
            for i in range(max_iters):

                # print("##### Iteration - ", i + 1)

                #### Eval  ######
                if i % eval_interval == 0 or i == max_iters - 1:
                    losses = estimate_loss_summary(model, batch_size, mini_batch_size, device, eval_iters, val_train_dataloader, val_val_dataloader, tokenizer)
                    print(
                        f"step {i}: train loss {losses['train']:.4f}, train_rouge {losses['train_rouge']:.4f}, val loss {losses['val']:.4f}, val_rouge {losses['val_rouge']:.4f}"
                    )
                    mlflow.log_metric("Train Loss", losses["train"], step=i)
                    mlflow.log_metric("Val Loss", losses["val"], step=i)
                    mlflow.log_metric("Train Rouge", losses["train_rouge"], step=i)
                    mlflow.log_metric("Val Rouge", losses["val_rouge"], step=i)

                    val_train_dataloader.reset()
                    with torch.no_grad():
                        xb, yb, mask = val_train_dataloader.next_batch()
                        # logits = model(xb, mask)
                        logits = model(xb)
                    # inputs = tokenizer.decode(xb[0], skip_special_tokens=False)
                    # targets = tokenizer.decode(yb[0], skip_special_tokens=False)

                    _ = cross_entropy_loss(logits, yb, tokenizer, tpye="train")
                    
                    top_k = 50
                    temperature = 0.7
                    # generate summary
                    # pos = [idx for idx, i in enumerate(xb[0]) if i == tokenizer.eos_token_id]
                    pos = [idx for idx, i in enumerate(xb[0]) if i == 25]
                    prompt = xb[0][:pos[1] + 1].unsqueeze(0)

                    # mask = torch.ones(1, len(prompt[0])).to(device)
                    while len(prompt[0]) <= 128:
                        # out = model(prompt, mask)
                        out = model(prompt)
                        
                        logits = out[:, -1, :]

                        if top_k > 0:
                            top_logits, _ = torch.topk(logits, top_k)
                            min_val = top_logits[:, -1]
                            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

                        if temperature > 0:
                            logits = logits / temperature

                            probs = torch.softmax(logits, dim=-1)

                            idx_next = torch.multinomial(probs, num_samples=1)

                        else:
                            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        prompt = torch.cat((prompt, idx_next), dim=1)
                        # mask = torch.cat((mask, torch.ones(1, 1).to(device)), dim=1)
                    
                    prompt = tokenizer.decode(prompt[0][pos[1]+1:].tolist())
                    print(f"AUTO: {repr(prompt)}\n") 


                ##### Train  ######
                # start_time = time.time()
                # set model to train mode
                model.train()

                # zero the gradients
                optimizer.zero_grad()

                #### Mini Batch  ######
                for _ in range(steps):

                    
                    xb, yb, mask = train_dataloader.next_batch()

                    # cast to bfloat16
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):

                        # get prediction
                        # logits = model(xb, mask)
                        logits = model(xb)
                        
                        # get loss
                        # rouge_loss = 1 - rouge_score(logits.argmax(dim=-1), yb, tokenizer)
                        ce_loss = cross_entropy_loss(logits, yb, tokenizer)
                        loss = ce_loss #+ rouge_loss

                    
                    # scale loss for accumulation
                    (loss / steps).backward()
                
                # clip and step after accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # print("##### Iteration - ", i + 1, " Time Taken - ", time.time() - start_time)
        
        # save the model
        mlflow.pytorch.log_model(
            model,
            f"{mlflow_run.info.run_id}",
            registered_model_name=registered_model_name,
        )

    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("training_summary_completion.txt", "w") as file:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    summary_training(config_path=parsed_args.config)
