import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution



class MyGCN(nn.Module):
    #(nfeat=3,nhid=8,nclass=16,dropout=0.5)
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(11016 * 16, 360)
        self.fc2 = nn.Linear(360, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.dropout = dropout

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute, solvent, solute_adj,solvent_adj):
        solute = F.relu(self.gc1(solute, solute_adj))
        solute = F.dropout(solute, self.dropout, training=self.training)
        solute = self.gc2(solute, solute_adj)

        solvent = F.relu(self.gc1(solvent, solvent_adj))
        solvent = F.dropout(solvent, self.dropout, training=self.training)
        solvent = self.gc2(solvent, solvent_adj)
        data = torch.cat((solute, solvent), 1)
        data = data.view(-1, self.num_flat_features(data))

        data = F.dropout(F.relu(self.fc1(data)), self.dropout, training=self.training)
        data = F.dropout(F.relu(self.fc2(data)),self.dropout, training=self.training)
        data = F.dropout(F.relu(self.fc3(data)),self.dropout, training=self.training)
        data = self.fc4(data)
        return data