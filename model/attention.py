import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, feat_dim):
        super(CoAttention, self).__init__()

    def forward(self, ifeat, qfeat, ifeats, qfeats):
        """
        Args:
            ifeat: N x 1 x D
            qfeat: N x 1 x D
            ifeats: N x F x D
            qfeats: N x L x D
        """
        iscores = torch.bmm(qfeat, ifeats.transpose(1, 2)) # N x 1 x F
        qscores = torch.bmm(ifeat, qfeats.transpose(1, 2)) # N x 1 x L
        ifeat = ifeat + torch.bmm(F.softmax(qscores, -1), qfeats) # N x 1 x D
        qfeat = qfeat + torch.bmm(F.softmax(iscores, -1), ifeats) # N x 1 x D

        return ifeat, qfeat


if __name__ == "__main__":
    ca = CoAttention(30)
    x = torch.rand(3, 1, 30)
    y = torch.rand(3, 1, 30)
    xs = torch.rand(3, 7, 30)
    ys = torch.rand(3, 5, 30)
    x, y = ca(x, y, xs, ys)
    print(x.size(), y.size())