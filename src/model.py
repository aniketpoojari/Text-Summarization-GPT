import torch
from torch.nn import functional as F
from data_loader import DataLoaderPretraining, DataLoaderSummary
import torch.nn as nn
from transformers import GPT2Tokenizer
from rouge_score import rouge_scorer
import inspect


def rouge_score(predictions, targets, tokenizer):

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_l_scores = []
    for prediction, target in zip(predictions, targets):
        # Remove Full Text
        mask = target != -1
        prediction = prediction[mask][:-1]
        target = target[mask][1:]

        prediction = tokenizer.decode(prediction, skip_special_tokens=True)
        target = tokenizer.decode(target, skip_special_tokens=True)

        score = scorer.score(target, prediction)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    # Averaging across all samples
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return avg_rouge_l


def estimate_loss_pretraining(
    model,
    batch_size,
    mini_batch_size,
    device,
    eval_iters,
    train_dataloader,
    val_dataloader,
):
    # set model to evaluation
    model.eval()

    # set precision
    torch.set_float32_matmul_precision("medium")

    # output dictionary
    out = {}

    # loop over train and val sets
    for split in ["train", "val"]:

        # set dataloader
        dataloader = train_dataloader if split == "train" else val_dataloader

        # reset dataloader
        dataloader.reset()

        # get number of steps
        steps = batch_size // mini_batch_size

        # initialize losses
        losses = torch.zeros(eval_iters)

        # loop over eval iterations
        for k in range(eval_iters):

            with torch.no_grad():

                # initialize batch loss
                batch_loss = 0

                # loop over steps
                for _ in range(steps):

                    # get batch
                    X, Y = dataloader.next_batch()

                    # set precision
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):

                        # forward
                        logits = model(X)

                        # reshape
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = Y.view(B * T)

                        # calculate loss
                        loss = F.cross_entropy(logits, targets)

                    # update batch loss
                    batch_loss += loss.detach().item()

                # update losses
                losses[k] = batch_loss / steps

        # update output
        out[split] = losses.mean()

    return out


def estimate_loss_summary(
    model,
    batch_size,
    mini_batch_size,
    device,
    eval_iters,
    val_train_dataloader,
    val_val_dataloader,
    tokenizer,
):
    # set model to evaluation
    model.eval()

    # set precision
    torch.set_float32_matmul_precision("medium")

    # output dictionary
    out = {}

    for split in ["train", "val"]:

        dataloader = val_train_dataloader if split == "train" else val_val_dataloader

        dataloader.reset()

        steps = batch_size // mini_batch_size

        losses = torch.zeros(eval_iters)
        rouge = torch.zeros(eval_iters)

        for k in range(eval_iters):

            with torch.no_grad():

                mini_batch_loss = 0
                mini_batch_rouge = 0

                for _ in range(steps):

                    X, Y = dataloader.next_batch()

                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits = model(X)

                        mini_batch_rouge += rouge_score(
                            logits.argmax(dim=-1), Y, tokenizer
                        )

                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        targets = Y.view(B * T)

                        valid_mask = targets != -1
                        logits = logits[valid_mask][:-1]
                        targets = targets[valid_mask][1:]

                        loss = F.cross_entropy(logits, targets)

                    mini_batch_loss += loss.detach().item()

                losses[k] = mini_batch_loss / steps
                rouge[k] = mini_batch_rouge / steps

        out[split] = losses.mean()
        out[split + "_rouge"] = rouge.mean()

    return out


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head):  # dropout:
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = n_head
        self.n_embd = n_embd

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(
        self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device
    ):
        super().__init__()

        self.n_layer = n_layer

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                h=nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.shape

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
