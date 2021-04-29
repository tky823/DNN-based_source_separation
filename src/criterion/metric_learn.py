import torch
import torch.nn as nn

EPS=1e-12

class TripletLoss(nn.Module):
    def __init__(self, dim=1, reduction='mean', margin=1, eps=EPS):
        super().__init__()

        self.dim = dim
        self.reduction = reduction
        self.margin = margin

        self.eps = eps
    
    def forward(self, anchor, positive, negative, batch_mean=True):
        assert positive.size() == negative.size(), "Invalid tensor size pair."

        loss_positive = torch.sum((positive - anchor)**2, dim=self.dim)
        loss_negative = torch.sum((negative - anchor)**2, dim=self.dim)

        n_dim = loss_positive.dim()
        
        if n_dim > 1:
            dim = range(1, n_dim)

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

class ConstrativeLoss(nn.Module):
    def __init__(self, margin=1, eps=EPS):
        super().__init__()

        self.margin = margin
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

        loss = is_same * distance**2 + (1 - is_same) * torch.relu(margin - distance) ** 2

        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss

def _test_triplet_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal
    from criterion.distance import L2Loss

    random.seed(111)
    torch.manual_seed(111)

    n_dim = 2
    batch_size = 4
    num_samples = batch_size
    
    mean1, mean2 = torch.ones(n_dim), - torch.ones(n_dim)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/tmp.png", bbox_inches='tight')
    
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


def _test_constrative_loss():
    import random

    import matplotlib.pyplot as plt

    from torch.distributions.multivariate_normal import MultivariateNormal
    from criterion.distance import L2Loss

    random.seed(111)
    torch.manual_seed(111)

    n_dim = 2
    batch_size = 4
    num_samples = batch_size
    
    mean1, mean2 = torch.ones(n_dim), - torch.ones(n_dim)
    covariance1, covariance2 = 2 * torch.tensor([[1.0, 0.5], [0.5, 1.0]]), 2 * torch.tensor([[1, 0.5], [0.5, 1]])
    m1, m2 = MultivariateNormal(mean1, covariance1), MultivariateNormal(mean2, covariance2)
    x1, x2 = m1.sample((num_samples,)), m2.sample((num_samples,))
    t1, t2 = torch.zeros(num_samples), torch.ones(num_samples)

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='red')
    plt.scatter(x2[:, 0], x2[:, 1], color='blue')
    plt.scatter(mean1[0], mean1[1], color='black', marker='x')
    plt.scatter(mean2[0], mean2[1], color='black', marker='x')
    plt.savefig("data/tmp.png", bbox_inches='tight')

    x = torch.cat([x1, x2], dim=0)
    t = torch.cat([t1, t2], dim=0)
    indices = list(range(2 * num_samples))
    random.shuffle(indices)
    x, t = x[indices].view(num_samples, 2, n_dim), t[indices].view(num_samples, 2)
    criterion = L2Loss()
    constrative_criterion = ConstrativeLoss()

    input, target = x[:batch_size], t[:batch_size]
    distance = criterion(input[:, 0], input[:, 1], batch_mean=False)
    is_same = (target[:, 0] == target[:, 1]).float()

    print(is_same)
    print(distance)

    loss = constrative_criterion(distance, is_same)
    
    print(loss)

if __name__ == '__main__':
    _test_triplet_loss()
    _test_constrative_loss()
