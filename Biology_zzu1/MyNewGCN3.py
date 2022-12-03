# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, Set2Set

from layers import GraphConvolution


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.linear_layer = nn.Linear(input_dim, output_dim)
        else:
            self.first_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            if num_layers > 2:
                self.middle_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU()
                    )
                    for _ in range(num_layers - 2)
                ])
            self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.num_layers == 1:
            x = self.linear_layer(x)
        else:
            x = self.first_layer(x)
            if self.num_layers > 2:
                for layer in self.middle_layers:
                    x = layer(x)
            x = self.last_layer(x)
        return x


# 新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):
    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyNewGCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.gc1 = GraphConv(nfeat, nhid, norm="both")
        self.gc2 = GraphConv(nhid, nclass, norm="both")

        self.gc3 = GraphConv(nfeat, nhid, norm="both")
        self.gc4 = GraphConv(nhid, nclass, norm="both")

        self.mlp_layers = MLP(
            2,
            9 if i == 0 else 16,
            hidden_dim,
            hidden_dim
        )

        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute, solvent, solute_len_matrix, solvent_len_matrix):
        solute_features1 = F.relu(self.gc1(solute, solute.ndata['x'].float()))
        solute_features = self.gc2(solute, solute_features1)

        solvent_features1 = F.relu(self.gc1(solvent, solvent.ndata['x'].float()))
        solvent_features = self.gc2(solvent, solvent_features1)


        solute_features1 = torch.mm(solute_len_matrix, solute_features1)  # (4,32)
        solute_features = torch.mm(solute_len_matrix, solute_features)  # (4,32)
        solvent_features1 = torch.mm(solvent_len_matrix, solvent_features1)  # (4,32)
        solvent_features = torch.mm(solvent_len_matrix, solvent_features)  # (4,32)
        solute_features = torch.cat((solute_features1, solute_features), 1)
        solvent_features = torch.cat((solvent_features1, solvent_features), 1)
        data = torch.cat((solute_features, solvent_features), 1)

        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = self.fc3(data)

        return data