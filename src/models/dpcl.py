import torch
import torch.nn as nn

from utils_model import get_num_parameters

EPS=1e-9

class Embedding(nn.Module):
    def __init__(self, F_bin, hidden_channels, num_layers, dimension, num_clusters, causal=False):
        """
        Default embedding: f_theta(x)
        """
        super(Embedding, self).__init__()
        
        self.F_bin, self.dimension = F_bin, dimension
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_clusters = num_clusters
        self.causal = causal
        
        if causal:
            num_directions = 1
            bidirectional = False
        else:
            num_directions = 2
            bidirectional = True
    
        self.num_directions = num_directions
            
        self.lstm = nn.LSTM(F_bin, hidden_channels, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(num_directions*hidden_channels, F_bin*dimension)
        self.nonlinear = nn.Sigmoid()
        
        self._get_num_parameters = get_num_parameters
        
        self.receptive_field, _ = self._get_receptive_field_params()
        self.num_parameters = self._get_num_parameters(self)
        
    def forward(self, input):
        """
        Args:
            input (bath_size, 1, F_bin, T_bin)
            output (batch_size, dimension, F_bin, T_bin)
        """
        hidden_channels, dimension = self.hidden_channels, self.dimension
        num_directions = self.num_directions
        batch_size, _, F_bin, T_bin = input.size()
        
        x = input.permute(0,1,3,2).contiguous() # -> (batch_size, _, T_bin, F_bin)
        x = x.view(batch_size, T_bin, F_bin)
        x, (_, _) = self.lstm(x) # -> (batch_size, T_bin, num_directions*hidden_channels)
        x = self.fc(x) # -> (batch_size, T_bin, F_bin*dimension)
        x = x.view(batch_size, T_bin, F_bin, dimension)
        x = x.permute(0,3,2,1).contiguous() # (batch_size, dimension, F_bin, T_bin)
        x = self.nonlinear(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        output = x / (norm+EPS)
        
        return output
        
    def _get_receptive_field_params(self, input_receptive_field=1, input_stride_product=1):
        return float("inf"), float("inf")
        
    @classmethod
    def build_model(cls, model_path):
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        model = cls(F_bin=package['F_bin'], hidden_channels=package['hidden_channels'], num_layers=package['num_layers'], dimension=package['dimension'], num_clusters=package['num_clusters'], causal=package['causal'])
        
        return model
        
    def get_package(self):
        package = {
            'F_bin': self.F_bin,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'dimension': self.dimension,
            'num_clusters': self.num_clusters,
            'causal': self.causal
        }
        
        return package
        
        
if __name__ == '__main__':
    from criterion import AffinityLoss
    
    batch_size, in_channels, F_bin, T_bin = 2, 1, 16, 64
    D = 10
    hidden_channels, num_layers = 20, 2
    C = 2
    causal = True
    
    input = torch.randn(batch_size, in_channels, F_bin, T_bin)
    
    target = torch.randn(batch_size, C, F_bin, T_bin)
    target = target.permute(0,2,3,1).contiguous()
    flatten_target = target.view(batch_size*F_bin*T_bin, C)
    flatten_idx = torch.arange(0, batch_size*F_bin*T_bin*C, C)
    flatten_idx = flatten_idx + flatten_target.argmax(dim=1)
    flatten_target = torch.zeros(batch_size*F_bin*T_bin*C)
    flatten_target[flatten_idx] = 1
    target = flatten_target.view(batch_size, F_bin, T_bin, C)
    target = target.permute(0,3,1,2).contiguous()
    
    model = Embedding(F_bin, hidden_channels=hidden_channels, num_layers=num_layers, dimension=D, num_clusters=C, causal=causal)
    print(model)
    criterion = AffinityLoss()
    
    output = model(input)
    print(output.size(), target.size())
    
    loss = criterion(output, target)
    print(loss)
    
    

