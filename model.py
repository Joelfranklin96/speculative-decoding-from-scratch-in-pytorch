"""
Speculative Decoding from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_corpus
import numpy as np

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
         "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon",
         "phi", "chi", "psi", "omega", "tensor", "kernel", "cache", "token", "batch", "decode",
         "prefill", "draft", "verify", "accept", "reject", "sample", "layer", "head", "logit", "stream"]

def make_corpus(n_sentences=400, seed=0):
    # TODO: successor table, then sentences from the chain; return one string
    rng = np.random.default_rng(seed)
    hashmap = {}
    for word in WORDS:
        hashmap[word] = rng.choice(len(WORDS), size=3, replace=False)

    ans_string = ""
    for i in range(n_sentences):
        sentence = ""
        last_word = WORDS[rng.integers(len(WORDS))]
        sentence += last_word
        l = rng.integers(5, 10)
        for j in range(l):
            if rng.random() < 0.9:
                last_word = WORDS[hashmap[last_word][rng.integers(3)]]
            else:
                last_word = WORDS[rng.integers(len(WORDS))]
            sentence += " " + last_word
        if i == n_sentences - 1:
            sentence += "."
        else:
            sentence += ". "
        ans_string += sentence
    return ans_string


class CharTokenizer:
    def __init__(self, text):
        # TODO: sorted character set, stoi/itos
        self.itos = list(sorted(set(text)))
        self.stoi = {s : i for i,s in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])

    @property
    def vocab(self):
        return len(self.itos)

# Step 2 - TinyGPT
import math
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        # TODO: ln1, ln2, qkv, proj, mlp, n_heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n_heads = n_heads

    def forward(self, x, kv=None):
        # TODO: pre-norm attention with cached k/v and an absolute-position causal mask, then the MLP
        B, T, d = x.shape
        x_norm = self.ln1(x)
        qkv = self.qkv(x_norm)
        d_head = d // self.n_heads
        if kv is None:
            q, k, v = torch.split(qkv, d, dim=-1)
            q = q.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
            k = k.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
            v = v.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
        else:
            q, k_new, v_new = torch.split(qkv, d, dim=-1)
            k_new = k_new.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
            v_new = v_new.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
            k_old, v_old = kv
            k = torch.cat((k_old, k_new), dim=2)
            v = torch.cat((v_old, v_new), dim=2)
            q = q.reshape(B, T, self.n_heads, d_head).transpose(1, 2)
            
        raw_scores = torch.matmul(q, torch.transpose(k, 2, 3))/d_head**0.5
        if kv is None:
            mask = torch.tril(torch.ones(T, T)) == 0
            raw_scores = raw_scores.masked_fill(mask, float('-inf'))
        else:
            a1 = q.shape[2]
            a2 = kv[0].shape[2]
            mask = torch.tril(torch.ones(a1, a1 + a2), diagonal=a2) == 0
            raw_scores = raw_scores.masked_fill(mask, float('-inf'))
        attention_scores = torch.softmax(raw_scores, dim = -1)
        output = torch.matmul(attention_scores, v)
        output = output.transpose(1, 2).reshape(B, T, -1)
        attention_output = self.proj(output)
        layer_one_output = x + attention_output
        layer_two_output = layer_one_output + self.mlp(self.ln2(layer_one_output))
        return (layer_two_output, (k, v))

class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, max_len=512):
        super().__init__()
        # TODO: tok, pos, blocks, ln, head
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, cache=None, return_hidden=False):
        # TODO: positions offset by the cached length; return (logits, new_cache[, hidden])
        new_cache = []
        x = self.tok(idx) + self.pos(torch.arange(len(idx)))
        if cache is None:
            for block in self.blocks:
                (x, (k, v)) = block.forward(x, None)
                new_cache.append((k, v))
        else:
            for i, block in enumerate(self.blocks):
                (x, (k, v)) = block.forward(x, kv=cache[i])
                new_cache.append((k,v))
        hidden = self.ln(x)
        logits = self.head(hidden)

        if return_hidden:
            return logits, new_cache, hidden
        else:
            return logits, new_cache

