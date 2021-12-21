"""
Modules for Transformer
"""
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout=0, max_len=5000, base=10000, batch_first=False):
        super().__init__()

        self.batch_first = batch_first

        position = torch.arange(max_len) # (max_len,)
        index = torch.arange(0, num_features, 2) / num_features # (num_features // 2,)        
        indices = position.unsqueeze(dim=1) / (base ** index.unsqueeze(dim=0)) # (max_len, num_features // 2)
        sin, cos = torch.sin(indices), torch.cos(indices)
        positional_encoding = torch.stack([sin, cos], dim=-1) # (max_len, num_features // 2, 2)

        if batch_first:
            positional_encoding = positional_encoding.view(max_len, num_features)
        else:
            positional_encoding = positional_encoding.view(max_len, 1, num_features)

        self.register_buffer("positional_encoding", positional_encoding)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        """
        Args:
            input: (T, batch_size, num_features) if batch_first=False, otherwise (batch_size, T, num_features)
        Returns:
            output: (T, batch_size, num_features) if batch_first=False, otherwise (batch_size, T, num_features)
        """
        if self.batch_first:
            T = input.size(1)
            x = input + self.positional_encoding[:, :T]
        else:
            T = input.size(0)
            x = input + self.positional_encoding[:T]

        output = self.dropout(x)

        return output

def _test():
    num_features = 64

    model = PositionalEncoding(num_features)

    input = torch.randn((32, 4, num_features))
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    _test()