import torch
import torch.nn.functional as F

EPS = 1e-12

class Word2Vec:
    def __init__(self, embedding_weights, vocab, eps=EPS):
        super().__init__()

        self.embedding_weights = embedding_weights
        self.vocab = vocab

        self.eps = eps

    def __call__(self, word_or_words, normalized=False):
        """
        Args:
            word_or_words <str> or <list<str>>: Word or words to vectorize.
        Returns:
            word_vec: Embedded word or words with shape of (embed_dim,) or (len(word_or_words), embed_dim).
        """
        if type(word_or_words) is str:
            word_idx = self.vocab[word_or_words] # <int>
        elif type(word_or_words) is list:
            word_idx = [self.vocab[word] for word in word_or_words] # (len(word_or_words),)
        else:
            raise TypeError("Not support {}.".format(type(word_or_words)))

        word_vec = self.embedding_weights[word_idx] # (embed_dim,) or # (len(word_or_words), embed_dim)

        if normalized:
            word_vec = F.normalize(word_vec, dim=-1, eps=self.eps) # (len(word_or_words), embed_dim)

        return word_vec

    def get_similar_words(self, word, k=10):
        """
        Args:
            word <str>: Word.
        Returns:
            similar_words <list<str>>
        """
        if type(word) is str:
            word_idx = self.vocab[word] # <int>
        else:
            raise TypeError("Not support {}.".format(type(word)))

        embedding_weights = self.embedding_weights
        normalized_embedding_weights = F.normalize(embedding_weights, dim=-1, eps=self.eps) # (vocab_size, embed_dim)

        if word_idx == 0:
            print("Out of vocabulary.")
            similar_words = [] # empty dictionary
        else:
            word_vec = normalized_embedding_weights[word_idx] # (embed_dim,)
            similar_words = self.get_similar_words_from_vec(word_vec, k=k+1)
            similar_words = similar_words[1:] # ignore word itself

        return similar_words

    def get_similar_words_from_vec(self, vec, k=10):
        """
        Args:
            vec: (embed_dim,)
        Returns:
            similar_words <list<str>>
        """
        eps = self.eps

        normalized_vec = F.normalize(vec, dim=-1, eps=eps) # (embed_dim,)
        embedding_weights = self.embedding_weights # (vocab_size, embed_dim)
        normalized_embedding_weights = F.normalize(embedding_weights, dim=-1, eps=self.eps) # (vocab_size, embed_dim)

        scores = torch.sum(normalized_embedding_weights * normalized_vec, dim=-1) # (vocab_size,)

        _, similar_words_idx = torch.topk(scores, k)
        similar_words_idx = similar_words_idx.tolist()
        similar_words = self.vocab.lookup_tokens(similar_words_idx)

        return similar_words