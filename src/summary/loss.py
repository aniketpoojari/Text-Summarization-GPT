import torch
from torch.nn import functional as F
from rouge_score import rouge_scorer
from dataloader import DataLoaderSummary

def rouge_score(predictions, targets, tokenizer):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = []
    for prediction, target in zip(predictions, targets):

        prediction = prediction[:-1]
        target = target[1:]

        # Remove -1 tokens
        mask = target != -1
        target = target[mask]
        prediction = prediction[mask]

        # Remove padding
        mask = target != tokenizer.pad_token_id
        target = target[mask]
        # mask = prediction != tokenizer.pad_token_id
        prediction = prediction[mask]
        
        # Decode full sequences properly
        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        target = tokenizer.decode(target, skip_special_tokens=True)

        # Calculate all Rouge metrics
        score = scorer.score(target=target, prediction=prediction)
        rouge_scores.append({
            'rouge1': score['rouge1'].fmeasure,
            'rouge2': score['rouge2'].fmeasure,
            'rougeL': score['rougeL'].fmeasure
        })
        

    # Average across all samples
    avg_rouge = {metric: sum(score[metric] for score in rouge_scores) / len(rouge_scores) for metric in ['rouge1', 'rouge2', 'rougeL']}

    return avg_rouge['rougeL']

def cross_entropy_loss(logits, Y, tokenizer, tpye=None):

    if tpye:
        pred = logits[0, :, :].argmax(dim=-1)
        tar = Y[0]
        # pred = pred[:-1]
        # tar = tar[1:]
        # mask = tar != tokenizer.pad_token_id
        # pred = pred[mask]
        # tar = tar[mask]
        pred = tokenizer.decode(pred.tolist())
        tar = tokenizer.decode(tar.tolist())
        print(f'\nPRED: {repr(pred)}\nTAR: {repr(tar)}')

    # logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    # Y = Y[:, 1:].contiguous().view(-1)
    logits = logits.contiguous().view(-1, logits.shape[-1])
    Y = Y.contiguous().view(-1)

    # loss = F.cross_entropy(logits, Y, reduction='mean', ignore_index=tokenizer.pad_token_id)
    loss = F.cross_entropy(logits, Y, reduction='mean', ignore_index=27156)
    return loss


def estimate_loss_summary(
    model,
    batch_size,
    mini_batch_size,
    device,
    eval_iters,
    train_dataloader,
    val_dataloader,
    tokenizer
):
    # set model to evaluation
    model.eval()

    # set precision
    torch.set_float32_matmul_precision("medium")

    # output dictionary
    out = {}

    for split in ["train", "val"]:

        dataloader = train_dataloader if split == "train" else val_dataloader

        steps = batch_size // mini_batch_size

        rouge = torch.zeros(eval_iters)
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            with torch.no_grad():

                batch_rouge = 0
                batch_loss = 0

                for _ in range(steps):

                    X, Y, mask = dataloader.next_batch()

                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        # logits = model(X, mask)
                        logits = model(X)

                        # mini_batch_rouge = rouge_score(logits.argmax(dim=-1), Y, tokenizer)
                        mini_batch_loss = cross_entropy_loss(logits, Y, tokenizer)

                    # batch_rouge += mini_batch_rouge
                    batch_loss += mini_batch_loss

                rouge[k] = batch_rouge / steps
                losses[k] = batch_loss / steps

        out[split + "_rouge"] = rouge.mean()
        out[split] = losses.mean()

    return out