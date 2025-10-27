# Based on Andrej Karpathy's nanoGPT

import torch
import torch.nn as nn
from torch.nn import functional as F
from selfatt import device


class Head(nn.Module):
    def __init__(self, head_size, n_embd, ds_kwargs, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) if ds_kwargs[
            'with_key'] else nn.Identity()
        self.query = nn.Linear(n_embd, head_size,
                               bias=False) if ds_kwargs['with_query'] else nn.Identity()
        self.value = nn.Linear(n_embd, head_size,
                               bias=False) if ds_kwargs['with_value'] else nn.Identity()
        block_size = ds_kwargs['block_size']
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def compute_variables(self, x, dropout=True):
        """
        We factor out the computations in the forward pass so we can extract the variables
         for plotting.
        """
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        wei = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        if dropout:  # We want to disable dropout for plotting
            wei = self.dropout(wei)
        V = self.value(x)
        out = wei @ V
        return K, Q, V, wei, out

    def forward(self, x):
        _, _, _, _, out = self.compute_variables(x)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, head_size, n_embd, ds_kwargs, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd=n_embd, ds_kwargs=ds_kwargs) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head, ds_kwargs):
        super().__init__()
        head_size = ds_kwargs['head_size']
        self.sa = MultiHeadAttention(n_head=n_head, head_size=head_size, n_embd=n_embd,
                                     ds_kwargs=ds_kwargs)
        self.ffwd = FeedFoward(n_embd) if ds_kwargs['ffwd'] else nn.Identity(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) if ds_kwargs[
            'block_with_layer_norm'] else nn.Identity(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) if ds_kwargs[
            'block_with_layer_norm'] else nn.Identity(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, n_layer, n_embd, n_head, vocab_size, ds_kwargs):
        super().__init__()
        self.block_size = ds_kwargs['block_size']
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head, ds_kwargs=ds_kwargs) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) if ds_kwargs[
            'with_layer_norm'] else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
