import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet50
from attention import CoAttention

class CoNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, feat_dim, output_dim, useco=False):
        super(CoNet, self).__init__()
        self.cnn = resnet50(feat_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, feat_dim//2, bidirectional=True)
        if useco:
            self.att = CoAttention(feat_dim)
        self.fc = nn.Linear(feat_dim, output_dim)
        self.useco = useco

    def forward(self, x, y):
        ifeat = self.cnn(x)
        qfeat = self.rnn(self.emb(y))
        if self.useco: # For ablation
            ifeat, qfeat = self.att(ifeat, qfeat)
        output = self.fc(ifeat*qfeat)
        return F.log_softmax(output)
