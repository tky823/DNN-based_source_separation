import torch
import torch.nn as nn

"""
    Reference: "Deep clustering: Discriminative embeddings for segmentation and separation"
    See https://arxiv.org/abs/1508.04306
"""

EPS = 1e-12

class AffinityLoss(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.maximize = False
        self.eps = eps

    def forward(self, input, target, binary_mask=None, batch_mean=True):
        """
        Args:
            input (batch_size, n_samples, embed_dim1)
            target (batch_size, n_samples, embed_dim2)
            binary_mask (batch_size, n_samples)
        Returns:
            loss () or (batch_size,)
        """
        eps = self.eps
        V, Y = input, target

        if binary_mask is not None:
            V, Y = binary_mask.unsqueeze(dim=-1) * V, binary_mask.unsqueeze(dim=-1) * Y

        trans_V, trans_Y = V.permute(0, 2, 1), Y.permute(0, 2, 1) # (batch_size, embed_dim1, n_samples), (batch_size, embed_dim2, n_samples)

        Ysum = Y.sum(dim=1, keepdim=True) # (batch_size, n_samples, 1)
        YY1 = torch.sum(Y * Ysum, dim=2) # (batch_size, n_samples)

        D = 1 / torch.sqrt(YY1 + eps) # (batch_size, n_samples)
        VD, YD = trans_V * D.unsqueeze(dim=1), trans_Y * D.unsqueeze(dim=1) # (batch_size, embed_dim1, n_samples), (batch_size, embed_dim2, n_samples)
        VDV, YDY = torch.bmm(VD, V), torch.bmm(YD, Y) # (batch_size, embed_dim1, embed_dim1), (batch_size, embed_dim2, embed_dim2)
        VDY = torch.bmm(VD, Y) # (batch_size, embed_dim, embed_dim2)

        loss = torch.sum(VDV**2, dim=(1, 2)) + torch.sum(YDY**2, dim=(1, 2)) - 2 * torch.sum(VDY**2, dim=(1, 2)) # (batch_size,)

        if batch_mean:
            loss = loss.mean(dim=0) # ()

        return loss

def _test_equality():
    batch_size, n_samples, embed_dim = 4, 12, 5

    Y = torch.randn(batch_size, n_samples, embed_dim)
    trans_Y = Y.permute(0, 2, 1) # (batch_size, embed_dim, n_samples)

    Ysum = Y.sum(dim=1, keepdim=True) # (batch_size, n_samples, 1)
    YY1_new = torch.sum(Y * Ysum, dim=2) # (batch_size, n_samples)

    YY = torch.bmm(Y, trans_Y) # (batch_size, n_samples, n_samples)
    YY1_old = YY.sum(dim=-1) # (batch_size, n_samples)

    print(YY1_new)
    print(YY1_old)

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_equality()