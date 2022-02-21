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
        neg_dot = torch.sum(input.unsqueeze(dim=1) * neg, dim=2) # (batch_size, num_neg_samples)
        loss_pos = - F.logsigmoid(pos_dot) # (batch_size,)
        loss_neg = - F.logsigmoid(-neg_dot) # (batch_size, num_neg_samples)
        loss = loss_pos + loss_neg.sum(dim=1) # (batch_size,)

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

class NegativeSamplingWithDistanceLoss:
    def __init__(self, distance_fn=None, activation="sigmoid", reduction="mean"):
        self.distance_fn = distance_fn
        self.activation = activation
        self.reduction = reduction

    def __call__(self, *args):
        """
        Args:
            input: (batch_size, embed_dim)
            pos:
                (batch_size, embed_dim) if self.distance_fn is given
                (batch_size,) if self.distance_fn is None
            neg:
                (batch_size, num_neg_samples, embed_dim) if self.distance_fn is given
                (batch_size, num_neg_samples) if self.distance_fn is None
        """
        if len(args) == 3:
            input, pos, neg = args
            input = input.unsqueeze(dim=1)
            pos = pos.unsqueeze(dim=1)
            distance_pos = self.distance_fn(input, pos) # (batch_size, 1)
            distance_neg = self.distance_fn(input, neg) # (batch_size, num_neg_samples)
        elif len(args) == 2:
            distance_pos, distance_neg = args # (batch_size,), (batch_size, num_neg_samples)
            distance_pos = distance_pos.unsqueeze(dim=1)
        else:
            raise NotImplementedError

        if self.activation == "sigmoid":
            loss_pos = - F.logsigmoid(-distance_pos) # (batch_size,)
            loss_neg = - F.logsigmoid(distance_neg) # (batch_size, num_neg_samples)
            loss = loss_pos.sum(dim=1) + loss_neg.sum(dim=1) # (batch_size,)
        else:
            distance = torch.cat([distance_pos, distance_neg], dim=1) # (batch_size, num_neg_samples + 1)
            target = torch.zeros(distance.size(0), dtype=torch.long)
            target = target.to(distance.device)
            loss = F.cross_entropy(- distance, target) # (batch_size)

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss