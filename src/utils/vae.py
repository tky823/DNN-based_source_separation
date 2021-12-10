import torch.nn as nn

class DataParallelWrapper(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)