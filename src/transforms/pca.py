import torch
import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, standardize=True):
        super().__init__()

        self.standardize = standardize
        self.std, self.mean = 1, 0
        self.proj_matrix = None

    def forward(self, data):
        """
        Args:
            data: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            output: (batch_size, num_samples, num_features) or (num_samples, num_features) in PCA domain.
        """
        n_dims = data.dim()

        assert n_dims in [2, 3], "data is expected 2D or 3D tensor."

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.standardize:
                self.mean, self.std = torch.mean(data, dim=1), torch.std(data, dim=1)
                standardized = self.preprocess(data)
            else:
                standardized = data

            cov = torch.bmm(standardized.permute(0, 2, 1), standardized) / standardized.size(1)
            _, proj_matrix = torch.linalg.eigh(cov) # Computed by assending order
            self.proj_matrix = torch.flip(proj_matrix, dims=(-1,))
        else:
            if self.proj_matrix is None:
                raise RuntimeError("proj_matrix is computed in advance.")

            if self.standardize:
                standardized = self.preprocess(data)
            else:
                standardized = data

        output = torch.bmm(standardized, self.proj_matrix)

        if n_dims == 2:
            output = output.squeeze(dim=0)

        return output

    def preprocess(self, input):
        return (input - self.mean.unsqueeze(dim=1)) / self.std.unsqueeze(dim=1)

def _test_pca():
    num_samples, num_features = 50, 5

    loc = torch.ones(num_features)
    cov = torch.tensor([
        [1, 0.8, 0, 0, 0],
        [0.8, 1, 0.2, 0, -0.5],
        [0, 0.2, 1, 0, 0],
        [-0.5, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], dtype=torch.float)

    sampler = torch.distributions.MultivariateNormal(loc, cov)
    data = sampler.sample((num_samples,))

    pca = PCA()
    pca.train()

    projected = pca(data)

    print(data.size())

    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1])
    plt.savefig("data/PCA/pca.png", bbox_inches='tight')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.manual_seed(111)

    _test_pca()