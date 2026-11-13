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

