import torch
from torch.nn import functional as F
from dataloader import DeepSpeedDataLoaderPretraining

def estimate_loss_pretraining(
    model,
    train_path,
    val_path,
    eval_batches,
    block_size,
    batch_size,
    mini_batch_size,
    device,
    rank,
    world_size
):

    model.eval()

    torch.set_float32_matmul_precision("medium")

    # output dictionary
    out = {}

    with torch.no_grad():

        # loop over train and val sets
        for split in ["train", "val"]:

            # Get specific dataloader path
            dataloader_path = train_path if split == "train" else val_path

            dataloader = DeepSpeedDataLoaderPretraining(
                batch_size=batch_size,
                seq_length=block_size,
                data_path=dataloader_path,
                device=device,
                rank=rank,
                world=world_size
            )
            
            # initialize losses for each eval batch
            losses = torch.zeros(eval_batches)

            # loop over eval batches
            for eval_batch in range(eval_batches):

                batch_loss = torch.tensor(0.0, device=device)
                
                # loop over mini-batche
                for _ in range(batch_size // mini_batch_size):
                   
                    X, Y = dataloader.next_batch()

                    with torch.autocast(device_type=device, dtype=torch.float16):
                        logits = model(X)
                        
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = Y.view(B * T)
                        
                        mini_batch_loss = F.cross_entropy(logits, targets)

                    # update total batch loss
                    batch_loss += mini_batch_loss

                # calculate average batch loss
                batch_loss /= (batch_size // mini_batch_size)

                # update losses for each eval batch
                losses[eval_batch] = batch_loss
        
            # update each split loss by average over eval batches
            out[split] = losses.mean()

    return out