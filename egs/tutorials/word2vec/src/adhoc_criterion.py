import torch
import torch.nn.functional as F

class NegativeSamplingLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, input, pos, neg):
        """
        Args:
            input: (batch_size, embed_dim)
            pos: (batch_size, embed_dim)
            neg: (batch_size, num_neg_samples, embed_dim)
        """
        pos_dot = torch.sum(input * pos, dim=1) # (batch_size,)
        neg_dot = torch.sum(input.unsqueeze(dim=1) * neg, dim=(1,2)) # (batch_size,)
        loss = - F.logsigmoid(pos_dot) - F.logsigmoid(-neg_dot) # (batch_size,)

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class NegativeSamplingWithDistanceLoss:
    def __init__(self, distance_fn=None, reduction="mean"):
        self.distance_fn = distance_fn
        self.reduction = reduction

    def __call__(self, input, pos, neg):
        """
        Args:
            input: (batch_size, embed_dim)
            pos: (batch_size, embed_dim)
            neg: (batch_size, num_neg_samples, embed_dim)
        """
        input = input.unsqueeze(dim=1)
        pos = pos.unsqueeze(dim=1)
        distance_pos = self.distance_fn(input, pos) # (batch_size, 1)
        distance_neg = self.distance_fn(input, neg) # (batch_size, num_neg_samples)
        distance_pos = distance_pos.sum(dim=-1) # (batch_size,)
        distance_neg = distance_neg.sum(dim=-1) # (batch_size,)
        loss = - F.logsigmoid(-distance_pos) - F.logsigmoid(distance_neg) # (batch_size,)

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss