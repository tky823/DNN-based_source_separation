from torchtext.vocab import build_vocab_from_iterator

def build_vocab(data_iter, tokenizer, min_freq=50):
    vocab = build_vocab_from_iterator(map(tokenizer, data_iter), specials=["<unk>"], min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])
    return vocab