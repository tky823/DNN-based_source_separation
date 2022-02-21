import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class TripletLoss(nn.Module):
    def __init__(self, dim=1, reduction='mean', margin=1, eps=EPS):
        super().__init__()

        self.dim = dim
        self.reduction = reduction
        self.maximize = False
        self.margin = margin

        self.eps = eps

    def forward(self, anchor, positive, negative, batch_mean=True):
        assert positive.size() == negative.size(), "Invalid tensor size pair."

        loss_positive = torch.sum((positive - anchor)**2, dim=self.dim)
        loss_negative = torch.sum((negative - anchor)**2, dim=self.dim)

        n_dims = loss_positive.dim()

        if n_dims > 1:
            dim = range(1, n_dims)

            if self.reduction == 'mean':
                loss_positive = loss_positive.mean(dim=dim)
                loss_negative = loss_negative.mean(dim=dim)
            elif self.reduction == 'sum':
                loss_positive = loss_positive.sum(dim=dim)
                loss_negative = loss_negative.sum(dim=dim)
            else:
                raise ValueError("Invalid reduction type")

        loss = torch.relu(loss_positive + self.margin - loss_negative)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class TripletWithDistanceLoss(nn.Module):
    def __init__(self, distance_fn=None, margin=1, eps=EPS):
        super().__init__()

        self.distance_fn = distance_fn
        self.margin = margin
        self.maximize = False
        self.eps = eps

        if self.distance_fn is None:
            raise ValueError("Specify `distance_fn`.")

    def forward(self, anchor, positive, negative, batch_mean=True):
        assert positive.size() == negative.size(), "Invalid tensor size pair"

        loss_positive = self.distance_fn(positive, anchor, batch_mean=False)
        loss_negative = self.distance_fn(negative, anchor, batch_mean=False)
        loss = torch.relu(loss_positive + self.margin - loss_negative)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1, eps=EPS):
        super().__init__()

        self.margin = margin
        self.maximize = False
        self.eps = eps

    def forward(self, distance, is_same, batch_mean=True):
        """
        Args:
            distance (batch_size, *)
            is_same (batch_size, *)
        Returns:
            loss () or (batch_size, )
        """
        margin = self.margin

        loss = is_same * (distance ** 2) + (1 - is_same) * (torch.relu(margin - distance) ** 2)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class ContrastiveWithDistanceLoss(nn.Module):
    def __init__(self, distance_fn=None, margin=1, eps=EPS):
        super().__init__()

        self.distance_fn = distance_fn
        self.margin = margin
        self.maximize = False
        self.eps = eps

        if self.distance_fn is None:
            raise ValueError("Specify `distance_fn`.")

    def forward(self, input_left, input_right, is_same, batch_mean=True):
        """
        Args:
            input_left (batch_size, *)
            input_right (batch_size, *)
            is_same (batch_size, *)
        Returns:
            loss () or (batch_size, )
        """
        margin = self.margin

        distance = self.distance_fn(input_left, input_right, batch_mean=False)
        loss = is_same * (distance ** 2) + (1 - is_same) * (torch.relu(margin - distance) ** 2)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class ImprovedTripletLoss(nn.Module):
    # TODO: implement here
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement `ImprovedTripletLoss`")

class AdaptedTripletLoss(nn.Module):
    # TODO: implement here
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement `AdaptedTripletLoss`")

class QuadrupletLoss(nn.Module):
    # TODO: implement here
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement `QuadrupletLoss`")

class AdditiveAngularMarginLoss(nn.Module):
    def __init__(self, scale=30.0, margin=0.5, easy_margin=False, eps=EPS):
        super().__init__()

        self.scale, self.margin = scale, margin
        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.cos_pi_m = - self.cos_m
        self.m_sin_pi_m = margin * self.sin_m

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input <torch.Tensor>: (batch_size, num_classes)
            target <torch.LongTensor>: (batch_size)
        Returns:
            output <torch.Tensor>: (batch_size,) or ()
        """

        """
        Addition theorem
        cos(phi) = cos(th + m)
                 = cos(th)cos(m) - sin(th)sin(m)
                 = cos(th) * cos(m) - sqrt(1 - cos(th)^2) * sin(m)
        cos(th) := y_pred
        """
        num_classes = input.size(-1)
        scale = self.scale
        cos_m, sin_m = self.cos_m, self.sin_m
        eps = self.eps

        cos_th = input
        sin_th = torch.sqrt(1 - cos_th**2 + eps)
        cos_phi = cos_th * cos_m - sin_th * sin_m # (batch_size, num_classes)

        # For target class
        if self.easy_margin:
            cos_phi = torch.where(cos_th < 0, cos_th, cos_phi) # (batch_size, num_classes)
        else:
            cos_phi = torch.where(cos_th > self.cos_pi_m, cos_th - self.m_sin_pi_m, cos_phi) # (batch_size, num_classes)
        
        # For non-target class
        mask = F.one_hot(target, num_classes=num_classes) # (batch_size, num_classes)
        input = scale * (mask * cos_phi + (1.0 - mask) * cos_th)
        loss = F.cross_entropy(input, target, reduction="none")

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

def _test_triplet_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal

    random.seed(111)
    torch.manual_seed(111)

    n_dims = 2
    batch_size = 4
    num_samples = batch_size

    mean1, mean2 = torch.ones(n_dims), - torch.ones(n_dims)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/distributions/triplet_loss.png", bbox_inches='tight')

    batch_anchor = []
    batch_positive, batch_negative = [], []
    for idx in range(batch_size):
        positive_class = random.randint(1, 2)

        idx_anchor = random.randint(0, num_samples - 1)
        idx_positive, idx_negative = random.randint(0, num_samples - 1), random.randint(0, num_samples - 1)

        if positive_class == 1:
            anchor = x1[idx_anchor]
            positive, nagative = x1[idx_positive], x2[idx_negative]
        else:
            anchor = x2[idx_anchor]
            positive, nagative = x2[idx_positive], x1[idx_negative]

        batch_anchor.append(anchor)
        batch_positive.append(positive)
        batch_negative.append(nagative)

    batch_anchor = torch.vstack(batch_anchor)
    batch_positive = torch.vstack(batch_positive)
    batch_negative = torch.vstack(batch_negative)

    criterion = TripletLoss()
    loss = criterion(batch_anchor, batch_positive, batch_negative, batch_mean=False)

    print(batch_anchor.size(), batch_positive.size(), batch_negative.size())
    print(loss)

def _test_triplet_with_distance_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal

    def distance_fn(input, target, batch_mean=True):
        loss = torch.sum((input - target)**2, dim=1)
        if batch_mean:
            loss = loss.mean()
        return loss

    random.seed(111)
    torch.manual_seed(111)

    n_dims = 2
    batch_size = 4
    num_samples = batch_size

    mean1, mean2 = torch.ones(n_dims), - torch.ones(n_dims)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/distributions/triplet_with_distance_loss.png", bbox_inches='tight')

    batch_anchor = []
    batch_positive, batch_negative = [], []
    for idx in range(batch_size):
        positive_class = random.randint(1, 2)

        idx_anchor = random.randint(0, num_samples - 1)
        idx_positive, idx_negative = random.randint(0, num_samples - 1), random.randint(0, num_samples - 1)

        if positive_class == 1:
            anchor = x1[idx_anchor]
            positive, nagative = x1[idx_positive], x2[idx_negative]
        else:
            anchor = x2[idx_anchor]
            positive, nagative = x2[idx_positive], x1[idx_negative]

        batch_anchor.append(anchor)
        batch_positive.append(positive)
        batch_negative.append(nagative)

    batch_anchor = torch.vstack(batch_anchor)
    batch_positive = torch.vstack(batch_positive)
    batch_negative = torch.vstack(batch_negative)

    triplet_criterion = TripletWithDistanceLoss(distance_fn=distance_fn)
    loss = triplet_criterion(batch_anchor, batch_positive, batch_negative, batch_mean=False)

    print(batch_anchor.size(), batch_positive.size(), batch_negative.size())
    print(loss)

def _test_contrastive_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal
    from criterion.distance import L2Loss

    random.seed(111)
    torch.manual_seed(111)

    n_dims = 2
    batch_size = 4
    num_samples = batch_size

    mean1, mean2 = torch.ones(n_dims), - torch.ones(n_dims)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))
    t1, t2 = torch.zeros(num_samples), torch.ones(num_samples)

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/distributions/contrastive_loss.png", bbox_inches='tight')

    x = torch.cat([x1, x2], dim=0)
    t = torch.cat([t1, t2], dim=0)
    indices = list(range(2 * num_samples))
    random.shuffle(indices)
    x, t = x[indices].view(num_samples, 2, n_dims), t[indices].view(num_samples, 2)
    criterion = L2Loss()
    contrastive_criterion = ContrastiveLoss()

    input, target = x[:batch_size], t[:batch_size]
    distance = criterion(input[:, 0], input[:, 1], batch_mean=False)
    is_same = (target[:, 0] == target[:, 1]).float()

    loss = contrastive_criterion(distance, is_same)

    print(is_same)
    print(distance)
    print(loss)

def _test_contrastive_with_distance_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal
    from criterion.distance import L2Loss

    random.seed(111)
    torch.manual_seed(111)

    n_dims = 2
    batch_size = 4
    num_samples = batch_size

    mean1, mean2 = torch.ones(n_dims), - torch.ones(n_dims)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))
    t1, t2 = torch.zeros(num_samples), torch.ones(num_samples)

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/distributions/contrastive_with_distance_loss.png", bbox_inches='tight')

    x = torch.cat([x1, x2], dim=0)
    t = torch.cat([t1, t2], dim=0)
    indices = list(range(2 * num_samples))
    random.shuffle(indices)
    x, t = x[indices].view(num_samples, 2, n_dims), t[indices].view(num_samples, 2)
    criterion = L2Loss()
    contrastive_criterion = ContrastiveWithDistanceLoss(distance_fn=criterion)

    input, target = x[:batch_size], t[:batch_size]
    is_same = (target[:, 0] == target[:, 1]).float()

    loss = contrastive_criterion(input[:, 0], input[:, 1], is_same)

    print(is_same)
    print(loss)

if __name__ == '__main__':
    print("="*10, "Triplet Loss", "="*10)
    _test_triplet_loss()
    print()

    print("="*10, "Triplet Loss", "="*10)
    _test_triplet_with_distance_loss()
    print()

    print("="*10, "Contrastive Loss", "="*10)
    _test_contrastive_loss()
    print()

    print("="*10, "Contrastive Loss", "="*10)
    _test_contrastive_with_distance_loss()
