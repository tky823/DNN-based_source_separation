import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class KMeansBase(nn.Module):
    def __init__(self, K=2, init_centroids='kmeans++'):
        super().__init__()

        self.K = K
        self.init_centroids = init_centroids

        self.cluster_ids, self.centroids = None, None

    def forward(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.cluster_ids is None or self.centroids is None:
                cluster_ids, centroids = self._init_kmeans(data)
                self.cluster_ids, self.centroids = cluster_ids, centroids
        else:
            raise NotImplementedError

        if n_dims == 2:
            cluster_ids = cluster_ids.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_ids, self.centroids = cluster_ids, centroids

        return cluster_ids

    def _init_kmeans(self, data):
        """
        Args:
            data: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        """
        K = self.K
        batch_size, num_samples, num_features = data.size()

        if self.init_centroids == 'kmeans++':
            centroid_ids = _init_kmeans_pp(data, K=K) # (batch_size, K)
        else:
            centroid_ids = _init_centroids_random(data, K=K) # (batch_size, K)

        shift = torch.arange(0, K * batch_size, K).long().to(centroid_ids.device) # (batch_size,)
        centroid_ids = centroid_ids + shift.unsqueeze(dim=1)
        flatten_centroid_ids = centroid_ids.view(batch_size * K) # (batch_size * K)
        flatten_data = data.view(batch_size * num_samples, num_features) # (batch_size * num_samples, num_features)
        flatten_centroids = flatten_data[flatten_centroid_ids]
        centroids = flatten_centroids.view(batch_size, K, num_features) # (batch_size, K, num_features)

        distance = _euclid_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids, centroids

class KMeans(KMeansBase):
    def __init__(self, K=2, init_centroids='kmeans++'):
        """
        Args:
            K <int>: number of clusters
        """
        super().__init__(K=K, init_centroids=init_centroids)

    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.cluster_ids is None or self.centroids is None:
                self.cluster_ids, self.centroids = self._init_kmeans(data)

            if iteration is not None:
                for idx in range(iteration):
                    cluster_ids, centroids = self.update_once(data, cluster_ids=self.cluster_ids, centroids=self.centroids)

                    self.cluster_ids, self.centroids = cluster_ids, centroids
            else:
                while True:
                    cluster_ids, centroids = self.update_once(data, cluster_ids=self.cluster_ids, centroids=self.centroids)
                    distance = _euclid_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.cluster_ids, self.centroids = cluster_ids, centroids

                    if distance == 0:
                        break
        else:
            cluster_ids = self.infer(data)
            centroids = self.centroids

        if n_dims == 2:
            cluster_ids = cluster_ids.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_ids, self.centroids = cluster_ids, centroids

        return self.cluster_ids

    def update_once(self, data, cluster_ids=None, centroids=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        Returns:
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        """
        K = self.K
        mask = F.one_hot(cluster_ids, num_classes=K) # (batch_size, num_samples, K)

        """
        1. Calculate centroids
        """
        masked_data = mask.unsqueeze(dim=3) * data.unsqueeze(dim=2) # (batch_size, num_samples, K, num_features)
        pseudo_centroids = masked_data.sum(dim=1) # (batch_size, K, num_features)
        denominator = mask.sum(dim=1).unsqueeze(dim=2) # (batch_size, K, 1)
        centroids = pseudo_centroids / denominator # (batch_size, K, num_features)

        """
        2. Put labels based on distance
        """
        distance = _euclid_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids, centroids

    def infer(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples)
        """
        centroids = self.centroids

        if centroids.dim() == 2:
            centroids = centroids.unsqueeze(dim=0) # Add batch dimension

        distance = _euclid_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids

class SoftKMeans(KMeansBase):
    """
    Reference: "Single-Channel Multi-Speaker Separation using Deep Clustering"
    """
    def __init__(self, K=2, alpha=1, init_centroids='kmeans++'):
        """
        Args:
            K <int>: number of clusters
            alpha <float>: Controls hardness of clustering
        """
        super().__init__(K=K, init_centroids=init_centroids)

        del self.cluster_ids

        self.cluster_probs = None
        self.alpha = alpha

    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        alpha = self.alpha
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.cluster_probs is None or self.centroids is None:
                _, centroids = self._init_kmeans(data)
                distance = data.unsqueeze(dim=2) - centroids.unsqueeze(dim=1) # (batch_size, num_samples, K, num_features)
                distance = torch.sum(distance**2, dim=3) # (batch_size, num_samples, K)
                cluster_probs = F.softmax(- alpha * distance, dim=2) # responsibility (batch_size, num_samples, K)
                self.cluster_probs, self.centroids = cluster_probs, centroids

            if iteration is not None:
                for idx in range(iteration):
                    cluster_probs, centroids = self.update_once(data, cluster_probs=cluster_probs, centroids=self.centroids)

                    self.centroids = centroids
            else:
                while True:
                    cluster_probs, centroids = self.update_once(data, cluster_probs=cluster_probs, centroids=self.centroids)
                    distance = _euclid_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.centroids = centroids

                    if distance == 0:
                        break
        else:
            cluster_probs = self.infer(data)
            centroids = self.centroids

        if n_dims == 2:
            cluster_probs = cluster_probs.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_probs, self.centroids = cluster_probs, centroids

        return self.cluster_probs

    def update_once(self, data, cluster_probs=None, centroids=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            cluster_probs: (batch_size, num_samples, K)
            centroids: (batch_size, K, num_features)
        Returns:
            cluster_probs: (batch_size, num_samples, K)
            centroids: (batch_size, K, num_features)
        """
        alpha = self.alpha

        """
        1. Calculate centroids
        """
        masked_data = cluster_probs.unsqueeze(dim=3) * data.unsqueeze(dim=2) # (batch_size, num_samples, K, num_features)
        pseudo_centroids = masked_data.sum(dim=1) # (batch_size, K, num_features)
        denominator = cluster_probs.sum(dim=1).unsqueeze(dim=2) # (batch_size, K, 1)
        centroids = pseudo_centroids / denominator # (batch_size, K, num_features)

        distance = data.unsqueeze(dim=2) - centroids.unsqueeze(dim=1) # (batch_size, num_samples, K, num_features)
        distance = torch.sum(distance**2, dim=3) # (batch_size, num_samples, K)
        cluster_probs = F.softmax(- alpha * distance, dim=2) # responsibility (batch_size, num_samples, K)

        return cluster_probs, centroids

    def infer(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_probs <torch.Tensor>: (batch_size, num_samples, K)
        """
        alpha = self.alpha
        centroids = self.centroids

        if centroids.dim() == 2:
            centroids = centroids.unsqueeze(dim=0) # Add batch dimension

        distance = data.unsqueeze(dim=2) - centroids.unsqueeze(dim=1) # (batch_size, num_samples, K, num_features)
        distance = torch.sum(distance**2, dim=3) # (batch_size, num_samples, K)
        cluster_probs = F.softmax(- alpha * distance, dim=2) # responsibility (batch_size, num_samples, K)

        return cluster_probs

"""
Spherical KMeans algorithm
    Reference: "Efficient clustering of very large document collections"
    See https://link.springer.com/chapter/10.1007/978-1-4615-1733-7_20
"""
class SphericalKMeans(KMeansBase):
    def __init__(self, K=2, init_centroids='kmeans++'):
        """
        Args:
            K <int>: number of clusters
        """
        super().__init__(K=K, init_centroids=init_centroids)

    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        normalized_data = data / torch.norm(data, dim=2, keepdim=True)

        if self.training:
            if self.cluster_ids is None or self.centroids is None:
                cluster_ids, centroids = self._init_kmeans(normalized_data)
                self.cluster_ids, self.centroids = cluster_ids, centroids

            if iteration is not None:
                for idx in range(iteration):
                    self.cluster_ids, self.centroids = self.update_once(normalized_data, cluster_ids=self.cluster_ids, centroids=self.centroids)
            else:
                while True:
                    cluster_ids, centroids = self.update_once(normalized_data, cluster_ids=self.cluster_ids, centroids=self.centroids)
                    distance = _euclid_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.cluster_ids, self.centroids = cluster_ids, centroids

                    if distance == 0:
                        break
        else:
            cluster_ids = self.infer(normalized_data)
            centroids = self.centroids

        if n_dims == 2:
            cluster_ids = cluster_ids.squeeze(dim=0) # (num_samples,)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

            self.cluster_ids, self.centroids = cluster_ids, centroids

        return self.cluster_ids

    def update_once(self, normalized_data, cluster_ids=None, centroids=None):
        """
        Args:
            normalized_data: (batch_size, num_samples, num_features)
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        Returns:
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        """
        K = self.K
        mask = F.one_hot(cluster_ids, num_classes=K) # (batch_size, num_samples, K)

        """
        1. Calculate centroids
        """
        masked_data = mask.unsqueeze(dim=3) * normalized_data.unsqueeze(dim=2) # (batch_size, num_samples, K, num_features)
        pseudo_centroids = masked_data.sum(dim=1) # (batch_size, K, num_features)
        denominator = mask.sum(dim=1).unsqueeze(dim=2) # (batch_size, K, 1)
        centroids = pseudo_centroids / denominator # (batch_size, K, num_features)
        centroids = centroids / torch.norm(centroids, dim=2, keepdim=True)

        """
        2. Put labels based on distance
        """
        distance = _neg_dot_product(normalized_data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids, centroids

    def infer(self, normalized_data):
        """
        Args:
            normalized_data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples)
        """
        centroids = self.centroids

        if centroids.dim() == 2:
            centroids = centroids.unsqueeze(dim=0) # Add batch dimension

        distance = _neg_dot_product(normalized_data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        return cluster_ids

    def _init_kmeans(self, normalized_data):
        """
        Args:
            normalized_data: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids: (batch_size, num_samples)
            centroids: (batch_size, K, num_features)
        """
        cluster_ids, centroids = super()._init_kmeans(normalized_data)
        centroids = centroids / torch.norm(centroids, dim=2, keepdim=True)

        return cluster_ids, centroids

class GMMCluteringBase(nn.Module):
    def __init__(self, K=2, init_centroids='kmeans++', init_kmeans=True, diag_cov=False, tol=1e-5, eps=EPS):
        super().__init__()

        self.K = K

        self.init_centroids, self.init_kmeans = init_centroids, init_kmeans
        self.diag_cov = diag_cov

        self.centroids, self.cov_matrix, self.mix_coeff = None, None, None
        self.cluster_probs = None

        if self.init_kmeans:
            self.kmeans = KMeans(K=K, init_centroids=init_centroids)

        self.tol = tol
        self.eps = eps

    def forward(self, data, centroids=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
            centroids <torch.Tensor>: (batch_size, K, num_features) or (K, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples)
            centroids <torch.Tensor>: (batch_size, K, num_features) or (K, num_features)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.centroids is None or self.mix_coeff is None or self.mix_coeff is None:
                centroids, cov_matrix, mix_coeff = self._init_GMM(data)
                self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff
        else:
            raise NotImplementedError

        cluster_probs = None

        if n_dims == 2:
            cluster_probs = cluster_probs.squeeze(dim=0) # (num_samples, K)
            centroids = centroids.squeeze(dim=0) # (K, num_features)

        self.cluster_probs, self.centroids = cluster_probs, centroids
        cluster_ids = torch.argmax(cluster_probs, dim=-2) # (num_samples,)

        return cluster_ids

    def _init_GMM(self, data):
        K = self.K
        batch_size, num_samples, num_features = data.size()

        if self.init_kmeans:
            assert self.init_centroids == self.kmeans.init_centroids, "Invalid init_centroids is specified."
            _ = self.kmeans(data)
            centroids = self.kmeans.centroids
        else:
            if self.init_centroids == 'kmeans++':
                centroid_ids = _init_kmeans_pp(data, K=K) # (batch_size, K)
            else:
                centroid_ids = _init_centroids_random(data, K=K) # (batch_size, K)

            shift = torch.arange(0, K * batch_size, K).long().to(centroid_ids.device) # (batch_size,)
            centroid_ids = centroid_ids + shift.unsqueeze(dim=1)
            flatten_centroid_ids = centroid_ids.view(batch_size * K) # (batch_size * K)
            flatten_data = data.view(batch_size * num_samples, num_features) # (batch_size * num_samples, num_features)
            flatten_centroids = flatten_data[flatten_centroid_ids]
            centroids = flatten_centroids.view(batch_size, K, num_features) # (batch_size, K, num_features)

        cov_matrix, mix_coeff = torch.eye(num_features), torch.ones(K) / K # (num_features, num_features), (K,)
        cov_matrix, mix_coeff = torch.tile(cov_matrix, (batch_size, K, 1, 1)), torch.tile(mix_coeff, (batch_size, 1)) # (batch_size, K, num_features, num_features), (batch_size, K)
        cov_matrix, mix_coeff = cov_matrix.to(data.device), mix_coeff.to(data.device)

        return centroids, cov_matrix, mix_coeff

    def compute_prob(self, data, centroids=None, cov_matrix=None, mix_coeff=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            centroids: (batch_size, K, num_features)
            cov_matrix: (batch_size, K, num_features, num_features)
            mix_coeff: (batch_size, K)
        Returns:
            prob: (batch_size, num_samples)
        """
        batch_size, num_samples, num_features = data.size()
        K = centroids.size(1)
        eps = self.eps

        x = data.unsqueeze(dim=1) - centroids.unsqueeze(dim=2) # (batch_size, K, num_samples, num_features)
        x = x.view(batch_size * K, num_samples, num_features) # (batch_size * K, num_samples, num_features)

        eye = torch.eye(num_features).to(cov_matrix.device) # (num_features, num_features)
        cov_matrix = cov_matrix + eps * eye
        precision_matrix = torch.linalg.inv(cov_matrix) # (batch_size, K, num_features, num_features)
        precision_matrix = precision_matrix.view(batch_size * K, num_features, num_features)
        det = torch.linalg.det(cov_matrix) # (batch_size, K)
        xSigma = torch.bmm(x, precision_matrix) # (batch_size * K, num_samples, num_features)
        xSigmax = torch.sum(xSigma * x, dim=2) # (batch_size * K, num_samples)

        numerator = torch.exp(-0.5 * xSigmax.view(batch_size, K, num_samples)) # (batch_size, K, num_samples)
        denominator = 2 * math.pi * torch.sqrt(det) # (batch_size, K)
        prob = numerator / denominator.unsqueeze(dim=2) # (batch_size, K, num_samples)
        prob = torch.sum(mix_coeff.unsqueeze(dim=2) * prob, dim=1) # (batch_size, num_samples)

        return prob

    def compute_responsibility(self, data, centroids=None, cov_matrix=None, mix_coeff=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            centroids: (batch_size, K, num_features)
            cov_matrix: (batch_size, K, num_features, num_features)
            mix_coeff: (batch_size, K)
        Returns:
            prob: (batch_size, K, num_samples)
        """
        batch_size, num_samples, num_features = data.size()
        K = centroids.size(1)
        eps = self.eps

        x = data.unsqueeze(dim=1) - centroids.unsqueeze(dim=2) # (batch_size, K, num_samples, num_features)
        x = x.view(batch_size * K, num_samples, num_features) # (batch_size * K, num_samples, num_features)

        eye = torch.eye(num_features).to(cov_matrix.device) # (num_features, num_features)
        cov_matrix = cov_matrix + eps * eye
        precision_matrix = torch.linalg.inv(cov_matrix) # (batch_size, K, num_features, num_features)
        precision_matrix = precision_matrix.view(batch_size * K, num_features, num_features)
        det = torch.linalg.det(cov_matrix) # (batch_size, K)
        xSigma = torch.bmm(x, precision_matrix) # (batch_size * K, num_samples, num_features)
        xSigmax = torch.sum(xSigma * x, dim=2) # (batch_size * K, num_samples)

        numerator = torch.exp(-0.5 * xSigmax.view(batch_size, K, num_samples)) # (batch_size, K, num_samples)
        denominator = 2 * math.pi * torch.sqrt(det) # (batch_size, K)
        prob = numerator / denominator.unsqueeze(dim=2) # (batch_size, K, num_samples)
        prob = mix_coeff.unsqueeze(dim=2) * prob # (batch_size, K, num_samples)
        prob = prob / torch.sum(prob, dim=1, keepdim=True) # (batch_size, K, num_samples)

        return prob

class GMMClustering(GMMCluteringBase):
    """
        Clustering based on Guassian Mixture Model.
    """
    def __init__(self, K=2, init_centroids='kmeans++', init_kmeans=True, diag_cov=False, tol=1e-5):
        super().__init__(K=K, init_centroids=init_centroids, init_kmeans=init_kmeans, diag_cov=diag_cov, tol=tol)

    def forward(self, data, iteration=None):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features) or (num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples) or (num_samples,)
        """
        n_dims = data.dim()

        if n_dims == 2:
            data = data.unsqueeze(dim=0) # (batch_size, num_samples, num_features), where batch_size = 1.

        if self.training:
            if self.centroids is None or self.cov_matrix is None or self.mix_coeff is None:
                self.centroids, self.cov_matrix, self.mix_coeff = self._init_GMM(data)

            self.cluster_probs = self.compute_responsibility(data, self.centroids, cov_matrix=self.cov_matrix, mix_coeff=self.mix_coeff)

            if iteration is not None:
                for idx in range(iteration):
                    cluster_probs, centroids, cov_matrix, mix_coeff = self.update_once(data, cluster_probs=self.cluster_probs, centroids=self.centroids, cov_matrix=self.cov_matrix, mix_coeff=self.mix_coeff)

                    self.cluster_probs = cluster_probs
                    self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff
            else:
                while True:
                    cluster_probs, centroids, cov_matrix, mix_coeff = self.update_once(data, cluster_probs=self.cluster_probs, centroids=self.centroids, cov_matrix=self.cov_matrix, mix_coeff=self.mix_coeff)
                    distance = _euclid_distance(self.centroids, centroids, dim=-1)
                    distance = distance.mean().item()

                    self.cluster_probs = cluster_probs
                    self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff

                    if distance < self.tol:
                        break
        else:
            cluster_ids = self.infer(data)
            centroids = self.centroids

        if n_dims == 2:
            cluster_probs = cluster_probs.squeeze(dim=0) # (num_samples,)
            centroids, cov_matrix, mix_coeff = centroids.squeeze(dim=0).squeeze(dim=0), mix_coeff.squeeze(dim=0) # (K, num_features), (K, num_features, num_features), (K,)
            self.cluster_probs = cluster_probs
            self.centroids, self.cov_matrix, self.mix_coeff = centroids, cov_matrix, mix_coeff

            cluster_ids = torch.argmax(self.cluster_probs, dim=1) # (num_samples,)
        else:
            cluster_ids = torch.argmax(self.cluster_probs, dim=2) # (batch_size, num_samples)

        return cluster_ids

    def update_once(self, data, cluster_probs, centroids=None, cov_matrix=None, mix_coeff=None):
        """
        Args:
            data: (batch_size, num_samples, num_features)
            cluster_prob: So called responsibility. (batch_size, K, num_samples)
            centroids: (batch_size, K, num_features)
            cov_matrix: (batch_size, K, num_features, num_features)
            mix_coeff: (batch_size, K)
        Returns:
            cluster_probs: (batch_size, K, num_samples)
            centroids: (batch_size, K, num_features)
            cov_matrix: (batch_size, K, num_features, num_features)
            mix_coeff: (batch_size, K)
        """
        batch_size, num_samples = data.size()[:2]
        K = centroids.size(1)

        """
        1. M-step
        """
        num_effective = cluster_probs.sum(dim=-1) # (batch_size, K)
        centroids = torch.sum(cluster_probs.unsqueeze(dim=3) * data.unsqueeze(dim=1), dim=2) / num_effective.unsqueeze(dim=-1) # (batch_size, K, num_features)
        x = data.unsqueeze(dim=1) - centroids.unsqueeze(dim=2) # (batch_size, K, num_samples, num_features)
        numerator = torch.sum(cluster_probs.view(batch_size, K, num_samples, 1, 1) * x.unsqueeze(dim=4) * x.unsqueeze(dim=3), dim=2) # (batch_size, K, num_features num_features)
        denominator = num_effective.unsqueeze(dim=-1) # (batch_size, K, 1, 1)
        cov_matrix = numerator / denominator # (batch_size, K, num_features, num_features)
        mix_coeff = num_effective / num_samples

        """
        2. E-step
        """
        cluster_probs = self.compute_responsibility(data, centroids, cov_matrix=cov_matrix, mix_coeff=mix_coeff) # (batch_size, K, num_samples)

        return cluster_probs, centroids, cov_matrix, mix_coeff

    def infer(self, data):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            cluster_ids <torch.Tensor>: (batch_size, num_samples)
        """
        raise NotImplementedError

def _euclid_distance(x, y, dim=-1):
    return torch.norm(x - y, dim=dim)

def _neg_dot_product(x, y, dim=-1):
    return - torch.sum(x * y, dim=dim)

def _init_centroids_random(data, K=2):
    """
    Args:
        data <torch.Tensor>: (batch_size, num_samples, num_features)
        K <int>: # of clusters
    Returns:
        centroid_ids <torch.LongTensor>: (batch_size, K)
    """
    batch_size, num_samples, _ = data.size()
    centroid_ids = []

    for _ in range(batch_size):
        _centroid_ids = torch.randperm(num_samples)[:K]
        centroid_ids.append(_centroid_ids)

    centroid_ids = torch.stack(centroid_ids, dim=0) # (batch_size, K)
    centroid_ids = centroid_ids.to(data.device)

    return centroid_ids

def _init_kmeans_pp(data, K=2, compute_distance=_euclid_distance):
    """
    Args:
        data <torch.Tensor>: (batch_size, num_samples, num_features)
    Returns:
        centroid_ids <torch.LongTensor>: (batch_size, K)
    """
    num_samples = data.size(1)

    centroid_ids = []

    for _data in data:
        _centroid_ids = torch.randperm(num_samples)[:1]
        _centroid_ids = _centroid_ids.to(_data.device)

        for _ in range(K - 1):
            centroids = _data[_centroid_ids] # (num_samples, num_features)
            distance = compute_distance(_data.unsqueeze(dim=1), centroids, dim=2) # (num_samples, K)
            distance, _ = torch.min(distance, dim=1)
            weights = distance / torch.sum(distance)
            _centroid_id = torch.multinomial(weights, 1) # Equals to categorical distribution.
            _centroid_ids = torch.cat([_centroid_ids, _centroid_id], dim=0)

        centroid_ids.append(_centroid_ids)

    centroid_ids = torch.stack(centroid_ids, dim=0)

    return centroid_ids

def _test_kmeans_pp_iteration():
    K = 2
    iteration = 10
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/KMeans/iteration/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/KMeans/iteration/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _ = kmeans(data, iteration=iteration) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, kmeans.centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/KMeans/iteration/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    # or same as ...
    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _ = kmeans(data, iteration=0) # Only initializes centroids.

    for idx in range(iteration):
        _ = kmeans(data, iteration=1)

        for batch_idx, (_data, _centroids) in enumerate(zip(data, kmeans.centroids)):
            plt.figure()
            x, y = torch.unbind(_data, dim=-1)
            plt.scatter(x, y, color='black')
            x, y = torch.unbind(_centroids, dim=-1)
            plt.scatter(x, y, color='red')
            plt.savefig("data/KMeans/iteration/{}/faithful-{}.png".format(batch_idx + 1, idx + 1), bbox_inches='tight')
            plt.close()

def _test_kmeans():
    K = 2
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/KMeans/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/KMeans/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    kmeans = KMeans(K=K)
    _ = kmeans(data) # (batch_size, num_samples)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, kmeans.centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/KMeans/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

def _test_soft_kmeans():
    K = 2
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/SoftKMeans/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/SoftKMeans/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    kmeans = SoftKMeans(K=K)
    _ = kmeans(data) # (batch_size, num_samples, K)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, kmeans.centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/SoftKMeans/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

def _test_spherical_kmeans():
    seed = 111
    torch.manual_seed(seed)

    colors = ["red", "blue", "green", "orange", "lightblue", "pink"]
    K = len(colors)
    N = 150

    loc = [[0, 0], [1, 1]]
    covariance_matrix = [[[1, 0], [0, 1]], [[2, -0.5], [-0.5, 2]]]
    data = []

    for _loc, _covariance_matrix in zip(loc, covariance_matrix):
        _loc = torch.tensor(_loc, dtype=float)
        _covariance_matrix = torch.tensor(_covariance_matrix, dtype=float)
        _sampler = torch.distributions.multivariate_normal.MultivariateNormal(_loc, covariance_matrix=_covariance_matrix)
        _data = _sampler.sample((N,))
        data.append(_data)

    data = torch.stack(data, dim=0)

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/SphericalKMeans/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/SphericalKMeans/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    kmeans = SphericalKMeans(K=K)
    cluster_ids = kmeans(data) # (batch_size,)

    for batch_idx, (_data, _cluster_ids, _centroids) in enumerate(zip(data, cluster_ids, kmeans.centroids)):
        plt.figure()

        for cluster_id in range(K):
            indices, = torch.nonzero(_cluster_ids == cluster_id, as_tuple=True)
            x, y = torch.unbind(_data[indices], dim=-1)
            plt.scatter(x, y, color=colors[cluster_id])

        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, marker="^", s=50, color="black")
        plt.savefig("data/SphericalKMeans/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

def _test_gmm_clustering():
    K = 2
    seed = 111

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data0 = torch.cat([x, y], dim=1)

    mat = torch.Tensor([[1, -0.1], [-0.1, 0.8]])
    data1 = torch.matmul(data0, mat)

    data = torch.stack([data0, data1], dim=0)
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)
    data = (data - mean) / std

    for batch_idx, _ in enumerate(data):
        os.makedirs("data/GMMClustering/None/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/GMMClustering/None/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    torch.manual_seed(seed)

    gmm_clustering = GMMClustering(K=K)
    _ = gmm_clustering(data) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, gmm_clustering.centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/GMMClustering/None/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    print("KMeans++, iteration")
    _test_kmeans_pp_iteration()
    print()

    print("KMeans")
    _test_kmeans()
    print()

    print("SoftKMeans")
    _test_soft_kmeans()
    print()

    print("Spehrical KMeans")
    _test_spherical_kmeans()
    print()

    print("GMM clusteing")
    # _test_gmm_clustering()