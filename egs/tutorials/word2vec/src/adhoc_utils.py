from collections import Counter

import torch
from torchtext.vocab import build_vocab_from_iterator

def build_vocab(data_iter, tokenizer, min_freq=50):
    vocab = build_vocab_from_iterator(map(tokenizer, data_iter), specials=["<unk>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])

    return vocab

def build_neg_freq(data_iter, vocab, tokenizer, smooth=0.75):
    counter = Counter()

    for idx, line in enumerate(data_iter):
        sentence = tokenizer(line)
        token_ids = vocab(sentence)
        counter.update(token_ids)

    assert len(vocab) == len(counter)

    neg_freq = []
    for idx in range(len(counter)):
        neg_freq.append(counter[idx]**smooth)

    return neg_freq

def build_neg_table(neg_freq):
    neg_table = []

    for idx, _neg_freq in enumerate(neg_freq):
        for _ in range(int(_neg_freq)):
            neg_table.append(idx)
    neg_table = torch.tensor(neg_table, dtype=torch.long)

    return neg_table