import random
import torch

class Kmeans:
    def __init__(self, data, K=2):
        """
        Args:
            data (num_data, dimension) <torch.Tensor>
        """
        self.K = K
        
        num_data = len(data)
        
        centroid_id = random.sample(range(num_data), K)
        centroids = data[centroid_id]
        
        data = data.unsqueeze(dim=1)
        distance = torch.norm((data - centroids), dim=2) # (num_data, K)
        distance, cluster_id = torch.min(distance, dim=1)
        
        self.onehot_labels = torch.eye(K)[cluster_id].to(data.device)
        self.num_data = num_data
        self.data = data
        
    def __call__(self, iteration=10):
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
        print(data.device, onehot_labels.device)
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

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    random.seed(111)

    K = 2
    iteration = 10

    df = pd.read_csv("data/faithful.csv")
    x, y = torch.Tensor(df['waiting']), torch.Tensor(df['eruptions'])
    data = torch.stack([x, y], axis=1)
    num_data = len(data)

    mean = data.mean(axis=0, keepdims=True)
    var = (data**2).mean(axis=0, keepdims=True)-mean**2
    data = (data - mean)/torch.sqrt(var)

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.savefig("data/faithful-0.png", bbox_inches='tight')

    kmeans = Kmeans(data, K=K)
    onehot_labels, centroids = kmeans(iteration=iteration) # (N, K), (K, D)

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centroids[:,0], centroids[:,1], c='red')
    plt.savefig("data/faithful-last.png", bbox_inches='tight')

    # or same as ...

    kmeans = Kmeans(data, K=K)
    
    for idx in range(iteration):
        onehot_labels, centroids = kmeans.update_once()
        
        plt.figure()
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(centroids[:,0], centroids[:,1], c='red')
        plt.savefig("data/faithful-{}.png".format(idx+1), bbox_inches='tight')

