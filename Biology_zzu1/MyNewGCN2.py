#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, Set2Set

from layers import GraphConvolution


#新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):
    #(nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyNewGCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.gc1 = GraphConv(nfeat, nhid, norm="both")
        self.gc2 = GraphConv(nhid, nclass, norm="both")

        self.gc3 = GraphConv(nfeat, nhid, norm="both")
        self.gc4 = GraphConv(nhid, nclass, norm="both")

        self.set2set= Set2Set(2 * nclass, 2, 1)


        self.fc1 = nn.Linear(4096, 1024)

        self.fc2 = nn.Linear(1024, 512)

        self.fc3 = nn.Linear(512, 1)

        self.dropout = dropout

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute, solvent, solute_len_matrix, solvent_len_matrix):

        solute_features = F.relu(self.gc1(solute, solute.ndata['x'].float()))
        solute_features = self.gc2(solute, solute_features)

        solvent_features = F.relu(self.gc3(solvent, solvent.ndata['x'].float()))
        solvent_features = self.gc4(solvent, solvent_features)
        # print("solute_features.shape=", solute_features.shape)  # (4, 2076, 16))
        # print("solvent_features.shape=", solvent_features.shape)  # (4, 8940, 16))

        len_map = torch.mm(solute_len_matrix.t(), solvent_len_matrix)
        interaction_map = torch.mm(solute_features, solvent_features.t())
        interaction_map = torch.tanh(interaction_map)
        interaction_map = torch.mul(len_map.float(), interaction_map)
        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        # print("solute_features.shape=", solute_features.shape)  # (4, 2076, 16))
        # print("solvent_features.shape=", solvent_features.shape)  # (4, 8940, 16))

        # solute_features = solute_features.reshape(4, -1, 16)
        # solvent_features = solvent_features.reshape(4, -1, 16)

        solute_features = self.set2set(solute, solute_features)  # (4, 32)
        solvent_features = self.set2set(solvent, solvent_features)  # (4, 32)


        data = torch.cat((solute_features, solvent_features), 1) # (4, 64)
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = self.fc3(data)
        return data