#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


#新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):
    #(nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, solute_solvent_size):
        super(MyNewGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(solute_solvent_size * 16, 360)
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

    def forward(self, solute, solvent, solute_adj, solvent_adj):
        solute = F.relu(self.gc1(solute, solute_adj))
        solute = self.gc2(solute, solute_adj)
        print("solute.shape=",solute.shape)#(4, 2076, 16))
        solvent = F.relu(self.gc1(solvent, solvent_adj))
        solvent = self.gc2(solvent, solvent_adj)
        print("solvent.shape=", solvent.shape)
        data = torch.cat((solute, solvent), 1)
        data = data.view(-1, self.num_flat_features(data)) #data1.shape=', (4, 176256))
        print("data1.shape=", data.shape)
        data = F.relu(self.fc1(data))
        print("data2.shape=", data.shape)
        data = F.relu(self.fc2(data))
        print("data3.shape=", data.shape)
        data = F.relu(self.fc3(data))
        print("data4.shape=", data.shape)
        data = self.fc4(data)
        return data