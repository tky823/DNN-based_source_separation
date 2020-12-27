class AffinityLoss(nn.Module):
    def __init__(self, ):
        super(AffinityLoss, self).__init__()
        
        self.maximize = False
        
    def forward(self, input_transposed, target_transposed, batch_mean=True):
        """
        Args:
            input_transposed (batch_size, D, F_bin, T_bin)
            target_transposed (batch_size, C, F_bin, T_bin)
            
            N = F_bin*T_bin
        """
        batch_size, D, F_bin, T_bin = input_transposed.size()
        batch_size, C, F_bin, T_bin = target_transposed.size()
        
        input_transposed = input_transposed.view(batch_size, D, F_bin*T_bin) # (batch_size, D, N)
        target_transposed = target_transposed.view(batch_size, C, F_bin*T_bin) # (batch_size, C, N)
        input = input_transposed.permute(0,2,1) # (batch_size, N, D)
        target = target_transposed.permute(0,2,1) # (batch_size, N, C)
        
        affinity_input = torch.matmul(input_transposed, input)
        affinity_target = torch.matmul(target_transposed, target)
        affinity_correlation = torch.matmul(input_transposed, target)
        
        affinity_input = affinity_input.view(batch_size, D*D)
        affinity_target = affinity_target.view(batch_size, C*C)
        affinity_correlation = affinity_correlation.view(batch_size, D*C)
        
        affinity_input = (affinity_input**2).sum(dim=1)
        affinity_correlation = (affinity_correlation**2).sum(dim=1)
        affinity_target = (affinity_target**2).sum(dim=1)
        
        loss = affinity_input - 2*affinity_correlation + affinity_target
        loss = loss.mean(dim=0)
        
        return loss