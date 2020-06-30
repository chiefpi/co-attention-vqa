import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, feat_dim):
        super(CoAttention, self).__init__()
        self.transform = 