import torch
from torch.nn import functional as F
import torch.nn as nn

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

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if attention_mask is not None:
            # Convert from [B, T] to [B, 1, 1, T] for broadcasting
            attention_mask = attention_mask.view(B, 1, 1, T)

            # Create a causal mask that also respects padding
            # First create a causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

            # Combine with padding mask (0 for padding tokens)
            # We need to broadcast the padding mask to match causal mask dimensions
            combined_mask = causal_mask * attention_mask

            # Use the mask with scaled_dot_product_attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                is_causal=False  # We're handling causality in our custom mask
            )
        else:
            # Use default causal mask
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


        y = (y.transpose(1, 2).contiguous().view(B, T, C))  # re-assemble all head outputs side by side
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

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
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

    def forward(self, idx, attention_mask=None):
        # idx is of shape (B, T)
        B, T = idx.shape

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, attention_mask)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits