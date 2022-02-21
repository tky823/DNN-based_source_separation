import random
import warnings

import torch

MINSCALE = 0.25
MAXSCALE = 1.25

def apply_random_flip(input, flip_rate=0.5, dim=0):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    if type(dim) is int:
        dim = (dim,)

    flip = random.random() < flip_rate

    if flip:
        output = torch.flip(input, dims=dim)
    else:
        output = input

    return output

class RandomFlip:
    def __init__(self, flip_rate=0.5, dim=0):
        self.flip_rate = flip_rate
        self.dim = dim

    def __call__(self, input):
        output = apply_random_flip(input, flip_rate=self.flip_rate, dim=self.dim)
        return output

class RandomScaling:
    def __init__(self, min=MINSCALE, max=MAXSCALE):
        warnings.warn("Use RandomGain instead.", DeprecationWarning)
        self.min, self.max = min, max

    def __call__(self, input):
        output = apply_random_gain(input, min=self.min, max=self.max)
        return output

def apply_random_gain(input, min=MINSCALE, max=MAXSCALE):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    scale = random.uniform(min, max)
    output = scale * input

    return output

class RandomGain:
    def __init__(self, min=MINSCALE, max=MAXSCALE):
        self.min, self.max = min, max

    def __call__(self, input):
        output = apply_random_gain(input, min=self.min, max=self.max)
        return output

def apply_random_sign(input, rate=0.5):
    """
    Args:
        input <torch.Tensot>: (*)
    Returns:
        output <torch.Tensot>: (*)
    """
    if random.random() < rate:
        sign = -1
    else:
        sign = 1

    output = sign * input

    return output

class RandomSign:
    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, input):
        output = apply_random_sign(input, rate=self.rate)
        return output