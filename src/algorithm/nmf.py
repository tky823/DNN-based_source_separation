import torch
import torch.nn as nn

__metrics__ = ['EU', 'KL', 'IS']

class NMF:
    def __init__(self, data, K=2, metric='EU'):
        """
        Args:
            data (F_bin, T_bin)
        """
        assert metric in __metrics__, "metric is expected any of {}, given {}".format(metric, __metrics__)
        
        self.K = K
        self.metric = metric
        self.data = data
        
        F_bin, T_bin = data.size()
        self.F_bin, self.T_bin = F_bin, T_bin
        
        self.basis = torch.rand(F_bin, K) + 1
        self.activation = torch.rand(K, T_bin) + 1
        
    def __call__(self, iteration=10):
        metric = self.metric
        
        for idx in range(iteration):
            basis, activation = self.update_once()
            
        return basis, activation
        
    def update_once(self):
        metric = self.metric
        F_bin, K, T_bin = self.F_bin, self.K, self.T_bin
        data, basis, activation = self.data, self.basis, self.activation
        
        basis_transpose = basis.permute(1,0)
        activation_transpose = activation.permute(1,0)
        
        if metric == 'EU':
            basis_next =  basis * torch.matmul(data, activation_transpose) / torch.matmul(basis, torch.matmul(activation, activation_transpose))
            activation_next = activation * torch.matmul(basis_transpose, data) / torch.matmul(torch.matmul(basis_transpose, basis), activation)
        elif metric == 'KL':
            raise NotImplementedError("Not support 'KL'(Kullback Leibler divergence)")
        elif metric == 'IS':
            raise NotImplementedError("Not support 'IS'(Itakura Saito divergence)")
        
        basis_next = torch.where(torch.isnan(basis_next), torch.zeros(F_bin, K), basis_next)
        activation_next = torch.where(torch.isnan(activation_next), torch.zeros(K, T_bin), activation_next)
    
        self.basis = basis_next
        self.activation = activation_next
        
        return basis_next, activation_next