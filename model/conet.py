import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import resnet50
from model.attention import CoAttention


class CoNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, output_dim, num_ifeats=64, feat_dim=2048, useco=False):
        super(CoNet, self).__init__()
        self.cnn = resnet50(num_ifeats)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, feat_dim//2, bidirectional=True, batch_first=True)
        if useco: # Use co-attention
            self.att = CoAttention(feat_dim)
        self.fc = nn.Linear(feat_dim, output_dim)
        self.useco = useco

    def forward(self, x, y):
        """
        Args:
            x: N x C x H x W
            y: N x L
        Returns:
            dist: N x O
        """
        ifeats = self.cnn(x) # N x F x D
        qfeats, _ = self.rnn(self.emb(y)) # N x L x D
        ifeat = torch.mean(ifeats, 1, keepdim=True) # N x 1 x D
        qfeat = torch.mean(qfeats, 1, keepdim=True) # N x 1 x D
        if self.useco: # For ablation
            ifeat, qfeat = self.att(ifeat, qfeat, ifeats, qfeats)
        output = self.fc(ifeat*qfeat)
        return F.log_softmax(output, -1).squeeze()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        print("Loaded model from file " + filename)


if __name__ == "__main__":
    net = CoNet(100, 200, 300, useco=True)
    x = torch.rand(4, 3, 50, 60)
    y = torch.LongTensor([[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6]])
    z = net(x, y)
    print(z.size())
