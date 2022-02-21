import torch

"""
References:
    Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry
"""
def lorentzian_scalar_product(x, y, dim=-1):
    """
    Args:
        x <torch.Tensor>: (*)
        y <torch.Tensor>: (*)
    Returns:
        product <torch.Tensor>: (*)
    """
    sections = [1, x.size(dim) - 1]
    x_split = torch.split(x, sections, dim=dim)
    y_split = torch.split(y, sections, dim=dim)
    product = - torch.sum(x_split[0] * y_split[0], dim=dim) + torch.sum(x_split[1] * y_split[1], dim=dim)

    return product