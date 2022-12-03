# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import GraphConvolution


# 新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):

    def get_graph_pool_ACE(self):
        temp_solute0 = torch.ones(1, 2076)
        temp_solute1 = torch.zeros(1, 2076)
        temp_solute2 = torch.cat((temp_solute1, temp_solute1), 1)
        temp_solute3 = torch.cat((temp_solute2, temp_solute1), 1)
        graph_pool_solute0 = torch.cat((temp_solute0, temp_solute3), 1)
        graph_pool_solute1 = torch.cat((torch.cat((temp_solute1, temp_solute0), 1), temp_solute2), 1)
        graph_pool_solute2 = torch.cat((torch.cat((temp_solute2, temp_solute0), 1), temp_solute1), 1)
        graph_pool_solute3 = torch.cat((temp_solute3, temp_solute0), 1)
        graph_pool_solute = torch.cat((graph_pool_solute0, graph_pool_solute1), 0)
        graph_pool_solute = torch.cat((graph_pool_solute, graph_pool_solute2), 0)
        graph_pool_solute = torch.cat((graph_pool_solute, graph_pool_solute3), 0)

        temp_solvent0 = torch.ones(1, 8940)
        temp_solvent1 = torch.zeros(1, 8940)
        temp_solvent2 = torch.cat((temp_solvent1, temp_solvent1), 1)
        temp_solvent3 = torch.cat((temp_solvent2, temp_solvent1), 1)
        graph_pool_solvent0 = torch.cat((temp_solvent0, temp_solvent3), 1)
        graph_pool_solvent1 = torch.cat((torch.cat((temp_solvent1, temp_solvent0), 1), temp_solvent2), 1)
        graph_pool_solvent2 = torch.cat((torch.cat((temp_solvent2, temp_solvent0), 1), temp_solvent1), 1)
        graph_pool_solvent3 = torch.cat((temp_solvent3, temp_solvent0), 1)
        graph_pool_solvent = torch.cat((graph_pool_solvent0, graph_pool_solvent1), 0)
        graph_pool_solvent = torch.cat((graph_pool_solvent, graph_pool_solvent2), 0)
        graph_pool_solvent = torch.cat((graph_pool_solvent, graph_pool_solvent3), 0)

        return graph_pool_solute, graph_pool_solvent

    def get_graph_pool_NMF(self):
        temp_solute0 = torch.ones(1, 2076)
        temp_solute1 = torch.zeros(1, 2076)
        temp_solute2 = torch.cat((temp_solute1, temp_solute1), 1)
        temp_solute3 = torch.cat((temp_solute2, temp_solute1), 1)
        graph_pool_solute0 = torch.cat((temp_solute0, temp_solute3), 1)
        graph_pool_solute1 = torch.cat((torch.cat((temp_solute1, temp_solute0), 1), temp_solute2), 1)
        graph_pool_solute2 = torch.cat((torch.cat((temp_solute2, temp_solute0), 1), temp_solute1), 1)
        graph_pool_solute3 = torch.cat((temp_solute3, temp_solute0), 1)
        graph_pool_solute = torch.cat((graph_pool_solute0, graph_pool_solute1), 0)
        graph_pool_solute = torch.cat((graph_pool_solute, graph_pool_solute2), 0)
        graph_pool_solute = torch.cat((graph_pool_solute, graph_pool_solute3), 0)

        temp_solvent0 = torch.ones(1, 12150)
        temp_solvent1 = torch.zeros(1, 12150)
        temp_solvent2 = torch.cat((temp_solvent1, temp_solvent1), 1)
        temp_solvent3 = torch.cat((temp_solvent2, temp_solvent1), 1)
        graph_pool_solvent0 = torch.cat((temp_solvent0, temp_solvent3), 1)
        graph_pool_solvent1 = torch.cat((torch.cat((temp_solvent1, temp_solvent0), 1), temp_solvent2), 1)
        graph_pool_solvent2 = torch.cat((torch.cat((temp_solvent2, temp_solvent0), 1), temp_solvent1), 1)
        graph_pool_solvent3 = torch.cat((temp_solvent3, temp_solvent0), 1)
        graph_pool_solvent = torch.cat((graph_pool_solvent0, graph_pool_solvent1), 0)
        graph_pool_solvent = torch.cat((graph_pool_solvent, graph_pool_solvent2), 0)
        graph_pool_solvent = torch.cat((graph_pool_solvent, graph_pool_solvent3), 0)

        return graph_pool_solute, graph_pool_solvent


    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, solute_solvent_size):
        super(MyNewGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

        self.dropout = dropout
        self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool_ACE()
        self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool_NMF()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF):
        graph_pool_solute_ACE = self.graph_pool_solute_ACE
        graph_pool_solvent_ACE = self.graph_pool_solvent_ACE
        graph_pool_solute_NMF = self.graph_pool_solute_NMF
        graph_pool_solvent_NMF = self.graph_pool_solvent_NMF

        solute_ACE = F.relu(self.gc1(solute_ACE, solute_adj))
        solute_ACE = self.gc2(solute_ACE, solute_adj)
        solute_NMF = F.relu(self.gc1(solute_NMF, solute_adj))
        solute_NMF = self.gc2(solute_NMF, solute_adj)

        # print("solute.shape=",solute.shape)#(4, 2076, 16))
        solvent_ACE = F.relu(self.gc1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.gc2(solvent_ACE, solvent_adj_ACE)
        solvent_NMF = F.relu(self.gc1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.gc2(solvent_NMF, solvent_adj_NMF)
        # print("solvent.shape=", solvent.shape)#(4, 8940, 16))

        solute_ACE = solute_ACE.reshape(-1, 16)
        solvent_ACE = solvent_ACE.reshape(-1, 16)
        solute_NMF = solute_NMF.reshape(-1, 16)
        solvent_NMF = solvent_NMF.reshape(-1, 16)

        # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)

        data1 = torch.cat((solute_ACE, solvent_ACE), 1)
        data1 = F.relu(self.fc1(data1))
        data1 = F.relu(self.fc2(data1))
        data1 = F.relu(self.fc3(data1))
        data1 = self.fc4(data1)

        data2 = torch.cat((solute_NMF, solvent_NMF), 1)
        data2 = F.relu(self.fc1(data2))
        data2 = F.relu(self.fc2(data2))
        data2 = F.relu(self.fc3(data2))
        data2 = self.fc4(data2)

        data = torch.cat((data1, data2), 0)
        return data