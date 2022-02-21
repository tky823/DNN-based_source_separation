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
    distr_table = []
    start_table = []
    count_table = []
    start = 0

    for idx, _neg_freq in enumerate(neg_freq):
        count = int(_neg_freq)
        for _ in range(count):
            distr_table.append(idx) # To avoid unseless memory consumption
        start_table.append(start)
        count_table.append(count)
        start += count

    distr_table = torch.tensor(distr_table, dtype=torch.long)
    start_table = torch.tensor(start_table, dtype=torch.long)
    count_table = torch.tensor(count_table, dtype=torch.long)
    
    neg_table = {
        "distr": distr_table,
        "start": start_table,
        "count": count_table
    }

    return neg_table