import random

import torch
import torch.nn as nn

class KMeansBase(nn.Module):
    def __init__(self, data, K=2, init_centroids='kmeans++'):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        """
        super().__init__()

        self.K = K
        
        batch_size, num_samples, num_features = data.size()
        
        if init_centroids == 'kmeans++':
            centroid_ids = self._init_kmeans_pp(data, K=K) # (batch_size, K)
        else:
            centroid_ids = self._init_kmeans_random(data, K=K) # (batch_size, K)
        
        centroid_ids  = centroid_ids.view(batch_size * K) # (batch_size * K)
        flatten_data = data.reshape(batch_size * num_samples, num_features) # (batch_size * num_samples, num_features)
        flatten_centroids = flatten_data[centroid_ids]
        centroids = flatten_centroids.view(batch_size, K, num_features) # (batch_size, K, num_features)
        
        distance = self.compute_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)
        
        self.cluster_ids = cluster_ids # (batch_size, num_samples)
        self.num_samples = num_samples
        self.data = data # (batch_size, num_samples, num_features)
    
    def _init_kmeans_random(self, data, K=2):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            centroid_ids <torch.LongTensor>: (batch_size, K)
        """
        batch_size, num_samples, _ = data.size()
        centroid_ids = []

        for _ in range(batch_size):
            _centroid_ids = random.sample(range(num_samples), K)
            centroid_ids.append(_centroid_ids)
        
        centroid_ids = torch.Tensor(centroid_ids).long() # (batch_size, K)
        centroid_ids = centroid_ids.to(data.device)

        return centroid_ids
    
    def _init_kmeans_pp(self, data, K=2):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
        Returns:
            centroid_ids <torch.LongTensor>: (batch_size, K)
        """
        _, num_samples, _ = data.size()

        centroid_ids = []

        for _data in data:
            _centroid_ids = random.choices(range(num_samples), k=1)

            for _ in range(K - 1):
                centroids = _data[_centroid_ids] # (num_samples, num_features)

                distance = self.compute_distance(_data.unsqueeze(dim=1), centroids, dim=2) # (num_samples, K)
                distance, _ = torch.min(distance, dim=1)
                weights = distance / torch.sum(distance)

                _centroid_ids += random.choices(range(num_samples), k=1, weights=weights)
            
            centroid_ids.append(_centroid_ids)
        
        centroid_ids = torch.Tensor(centroid_ids).long()
        centroid_ids = centroid_ids.to(data.device)

        return centroid_ids
    
    def compute_distance(self, x, y, dim=-1, keepdim=False):
        distance = torch.norm(x - y, dim=dim, keepdim=keepdim)

        return distance

class KMeans(KMeansBase):
    def __init__(self, data, K=2, init_centroids='kmeans++'):
        """
        Args:
            data <torch.Tensor>: (batch_size, num_samples, num_features)
            K <int>: number of clusters
        """
        super().__init__(data, K=K, init_centroids=init_centroids)
    
    def forward(self, iteration=10):
        assert iteration > 0, "iteration should be positive."
        
        for idx in range(iteration):
            cluster_ids, centroids = self.update_once()
        
        return cluster_ids, centroids
        
    def update_once(self):
        K = self.K
        cluster_ids = self.cluster_ids # (batch_size, num_samples)
        data = self.data # (batch_size, num_samples, num_features)

        mask = torch.eye(K)[cluster_ids] # (batch_size, num_samples, K)
        mask = mask.to(data.device)

        """
        1. Calculate centroids
        """
        masked_data = mask.unsqueeze(dim=3) * data.unsqueeze(dim=2) # (batch_size, num_samples, K, num_features)
        pseudo_centroids = masked_data.sum(dim=1) # (batch_size, K, num_features)
        denominator = mask.sum(dim=1).unsqueeze(dim=2) # (batch_size, K, 1)
        centroids = pseudo_centroids / denominator  # (batch_size, K, num_features)
        
        """
        2. Put labels based on distance
        """
        distance = self.compute_distance(data.unsqueeze(dim=2), centroids.unsqueeze(dim=1), dim=3) # (batch_size, num_samples, K)
        cluster_ids = torch.argmin(distance, dim=2) # (batch_size, num_samples)

        self.cluster_ids = cluster_ids
    
        return cluster_ids, centroids

def _test_kmeans():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

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
        os.makedirs("data/KMeans/{}".format(batch_idx + 1), exist_ok=True)

    for batch_idx, _data in enumerate(data):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        plt.savefig("data/KMeans/{}/faithful-0.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    random.seed(seed)
    torch.manual_seed(seed)

    kmeans = KMeans(data, K=K)
    _, centroids = kmeans(iteration=iteration) # (batch_size, K), (batch_size, K, num_features)

    for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
        plt.figure()
        x, y = torch.unbind(_data, dim=-1)
        plt.scatter(x, y, color='black')
        x, y = torch.unbind(_centroids, dim=-1)
        plt.scatter(x, y, color='red')
        plt.savefig("data/KMeans/{}/faithful-last.png".format(batch_idx + 1), bbox_inches='tight')
        plt.close()

    # or same as ...
    random.seed(seed)
    torch.manual_seed(seed)

    kmeans = KMeans(data, K=K)

    for idx in range(iteration):
        _, centroids = kmeans.update_once()

        for batch_idx, (_data, _centroids) in enumerate(zip(data, centroids)):
            plt.figure()
            x, y = torch.unbind(_data, dim=-1)
            plt.scatter(x, y, color='black')
            x, y = torch.unbind(_centroids, dim=-1)
            plt.scatter(x, y, color='red')
            plt.savefig("data/KMeans/{}/faithful-{}.png".format(batch_idx + 1, idx + 1), bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    _test_kmeans()

