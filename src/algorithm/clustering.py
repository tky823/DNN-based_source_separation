import random

import torch
import torch.nn as nn

class KMeans(nn.Module):
    def __init__(self, data, K=2, init_centroids='kmeans++'):
        """
        Args:
            data (num_data, dimension) <torch.Tensor>
        """
        super().__init__()

        self.K = K
        
        num_data = len(data)
        
        if init_centroids == 'kmeans++':
            centroid_id = self._init_kmeans_pp(data, K=K)
        else:
            centroid_id = random.sample(range(num_data), K)
        centroids = data[centroid_id]
        
        data = data.unsqueeze(dim=1)
        distance = torch.norm((data - centroids), dim=2) # (num_data, K)
        distance, cluster_id = torch.min(distance, dim=1)
        
        self.onehot_labels = torch.eye(K)[cluster_id].to(data.device)
        self.num_data = num_data
        self.data = data
    
    def _init_kmeans_pp(self, data, K=2):
        """
        Args:
            data (num_data, dimension) <torch.Tensor>
        Returns:
            centroid_id <list<int>>: where len(centroid_id) = K
        """
        num_data = len(data)
        centroid_id = random.choices(range(num_data), k=1)

        while len(centroid_id) < K:
            centroids = data[centroid_id]

            data = data.unsqueeze(dim=1)
            distance = torch.norm(data - centroids, dim=2) # (num_data, K)
            distance, _ = torch.min(distance, dim=1)
            distance = distance**2
            weights = distance / torch.sum(distance)

            centroid_id += random.choices(range(num_data), k=1, weights=weights)

        return centroid_id
        
    def __call__(self, iteration=10):
        onehot_labels, centroids = None, None
        
        for idx in range(iteration):
            onehot_labels, centroids = self.update_once()
        
        return onehot_labels, centroids
        
    def update_once(self):
        num_data, K = self.num_data, self.K
        onehot_labels = self.onehot_labels.view(num_data, K, 1) # (num_data, K, 1)
        data = self.data # (num_data, 1, D)
        
        """
        1. Calculate centroids
        """
        masked_data = data * onehot_labels # (num_data, K, D)
        pseudo_centroids = masked_data.sum(dim=0)
        normalizer = onehot_labels.sum(dim=0)
        centroids = pseudo_centroids / normalizer  # (K, D)
        
        """
        2. Put labels based on distance
        """
        distance = torch.norm((data - centroids), dim=2) # (num_data, K)
        distance, cluster_id = torch.min(distance, dim=1)
        onehot_labels = torch.eye(K)[cluster_id].to(data.device)
        
        self.num_data = num_data
        self.data = data
        self.onehot_labels = onehot_labels
    
        return onehot_labels, centroids

def _test_kmeans():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs("data/Kmeans", exist_ok=True)
    random.seed(111)

    K = 2
    iteration = 10

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']).unsqueeze(dim=1), torch.Tensor(df['eruptions']).unsqueeze(dim=1)
    data = torch.cat([x, y], dim=1)
    num_data = len(data)

    mean = data.mean(dim=0, keepdim=True)
    var = (data**2).mean(dim=0, keepdim=True)-mean**2
    data = (data - mean)/torch.sqrt(var)

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.savefig("data/Kmeans/faithful-0.png", bbox_inches='tight')

    kmeans = KMeans(data, K=K)
    onehot_labels, centroids = kmeans(iteration=iteration) # (N, K), (K, D)

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centroids[:,0], centroids[:,1], c='red')
    plt.savefig("data/Kmeans/faithful-last.png", bbox_inches='tight')

    # or same as ...

    kmeans = KMeans(data, K=K)
    
    for idx in range(iteration):
        onehot_labels, centroids = kmeans.update_once()
        
        plt.figure()
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(centroids[:,0], centroids[:,1], c='red')
        plt.savefig("data/Kmeans/faithful-{}.png".format(idx+1), bbox_inches='tight')


if __name__ == '__main__':
    _test_kmeans()

