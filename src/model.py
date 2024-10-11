import torch
from torch.nn import functional as F
from data_loader import get_batch
import torch.nn as nn
from transformers import GPT2Tokenizer
from rouge_score import rouge_scorer


def clean_and_decode(predictions, targets, tokenizer):

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_l_scores = []
    for prediction, target in zip(predictions, targets):
        # Remove Full Text
        mask = target != -1
        prediction = prediction[mask]
        target = target[mask]

        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        target = tokenizer.decode(target, skip_special_tokens=True)

        score = scorer.score(target, prediction)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    # Averaging across all samples
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return avg_rouge_l


def estimate_loss(
    model,
    step,
    eval_iters,
    block_size,
    batch_size,
    device,
    pre_train=None,
    pre_val=None,
    train_full=None,
    train_summ=None,
    val_full=None,
    val_summ=None,
):
    out = {}
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {"pad_token": "<PAD>", "sep_token": "<SEP>"}
    tokenizer.add_special_tokens(special_tokens_dict)

    model.eval()
    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)
        rouge = torch.zeros(eval_iters)
        for k in range(eval_iters):

            X, Y = get_batch(
                step,
                split,
                block_size,
                batch_size,
                device,
                pre_train,
                pre_val,
                train_full,
                train_summ,
                val_full,
                val_summ,
            )

            logits = model(X)

            if step == "summary":
                rouge_scorer = clean_and_decode(logits.argmax(dim=-1), Y, tokenizer)
                rouge[k] = rouge_scorer

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = Y.view(B * T)

            if step == "summary":
                valid_mask = targets != -1
                targets = targets[valid_mask][1:]
                logits = logits[valid_mask][:-1]

            loss = F.cross_entropy(logits, targets, ignore_index=50257)
            losses[k] = loss.item()

        out[split] = losses.mean()
        if step == "summary":
            out[split + "_rouge"] = rouge.mean()

    model.train()

    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        v = self.value(x)  # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_head, n_embd, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(
        self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_head, n_embd, dropout, block_size) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.device = device

    def forward(self, idx):
        _, T = idx.shape  # B, T

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits
