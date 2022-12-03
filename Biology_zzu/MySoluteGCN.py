import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution



class MySoluteGCN(nn.Module):
    #(nfeat=3,nhid=8,nclass=16,dropout=0.5)
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MySoluteGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x