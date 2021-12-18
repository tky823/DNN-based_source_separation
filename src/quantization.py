import torch
import torch.nn as nn

class QuantizationWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.quant, self.dequant = torch.quantization.QuantStub(), torch.quantization.DeQuantStub()

    def forward(self, input):
        x = self.quant(input)
        x = self.base_model(x)
        output = self.dequant(x)

        return output