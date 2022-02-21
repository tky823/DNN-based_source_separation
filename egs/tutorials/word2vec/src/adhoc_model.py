import torch
import torch.nn as nn

class CBoW(nn.Module):
    def __init__(self, vocab_size, embed_dim, bias=False, max_norm=1):
        super().__init__()

        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.max_norm = max_norm

        self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=max_norm)
        self.linear = nn.Linear(embed_dim, vocab_size, bias=bias)

    def forward(self, input):
        """
        Args:
            input: (batch_size, 2 * context_size)
        Returns:
            output: (batch_size, vocab_size)
        """
        x = self.embedding(input) # (batch_size, 2 * context_size, embed_dim)
        x = x.mean(dim=1)
        output = self.linear(x)

        return output

    def get_embedding_weights(self):
        """
        Returns:
            weights: (vocab_size, embed_dim)
        """
        max_norm = self.max_norm
        weights = self.embedding.weight.data

        if max_norm is not None:
            norm = torch.linalg.vector_norm(weights, dim=1, keepdim=True) # (vocab_size, 1)
            weights = torch.where(norm > max_norm, max_norm * weights / norm, weights)

        return weights

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim, bias=False, max_norm=1):
        super().__init__()

        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.max_norm = max_norm

        self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=max_norm)
        self.linear = nn.Linear(embed_dim, vocab_size, bias=bias)

    def forward(self, input):
        """
        Args:
            input: (batch_size,) or (batch_size, context_size)
        Returns:
            output: (batch_size, vocab_size) or (batch_size, vocab_size, context_size)
        """
        x = self.embedding(input)
        x = self.linear(x)

        if x.dim() == 3:
            output = x.permute(0, 2, 1)
        else:
            output = x

        return output

    def get_embedding_weights(self):
        """
        Returns:
            weights: (vocab_size, embed_dim)
        """
        max_norm = self.max_norm
        weights = self.embedding.weight.data

        if max_norm is not None:
            norm = torch.linalg.vector_norm(weights, dim=1, keepdim=True) # (vocab_size, 1)
            weights = torch.where(norm > max_norm, max_norm * weights / norm, weights)

        return weights

class CBoWNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_norm=1):
        super().__init__()

        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.max_norm = max_norm

        self.enc_embedding = nn.Embedding(vocab_size, embed_dim, max_norm=max_norm)
        self.dec_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input, pos=None, neg=None):
        """
        Args:
            input: (batch_size, 2 * context_size)
            pos: (batch_size,)
            neg: (batch_size, num_neg_samples)
        Returns:
            output: (batch_size, embed_dim)
            pos_output: (batch_size, embed_dim)
            neg_output: (batch_size, num_neg_samples, embed_dim)
        """
        output = self.enc_embedding(input)
        output = output.mean(dim=1)

        if pos is None and neg is None:
            return output

        if pos is None or neg is None:
            raise ValueError("Specify pos and neg.")

        pos_output = self.dec_embedding(pos) # (batch_size, embed_dim)
        neg_output = self.dec_embedding(neg) # (batch_size, num_neg_samples, embed_dim)

        return output, pos_output, neg_output

    def get_embedding_weights(self):
        """
        Returns:
            weights: (vocab_size, embed_dim)
        """
        max_norm = self.max_norm

        weights = self.enc_embedding.weight.data
        norm = torch.linalg.vector_norm(weights, dim=1, keepdim=True) # (vocab_size, 1)
        weights = torch.where(norm > max_norm, max_norm * weights / norm, weights)

        return weights