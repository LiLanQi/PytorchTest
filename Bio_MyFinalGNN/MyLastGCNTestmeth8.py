# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import GraphConvolution
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp

# 新的网络，可以容纳任何溶质
class MyValModel(nn.Module):

    def get_graph_pool(self, solvent):
        temp_solute0 = torch.ones(1, 2076)
        temp_solute1 = torch.zeros(1, 2076)
        for i in range(self.batch_size):
            temp_solute = temp_solute0
            for j in range(self.batch_size - i - 1):
                temp_solute = torch.cat((temp_solute, temp_solute1), 1)
            for j in range(i):
                temp_solute = torch.cat((temp_solute1, temp_solute), 1)
            if (i == 0):
                graph_pool_solute = temp_solute
            else:
                graph_pool_solute = torch.cat((graph_pool_solute, temp_solute), 0)
        if (solvent == "ACE"):
            temp_solvent0 = torch.ones(1, 8940)
            temp_solvent1 = torch.zeros(1, 8940)
        elif (solvent == "NMF"):
            temp_solvent0 = torch.ones(1, 12150)
            temp_solvent1 = torch.zeros(1, 12150)
        elif (solvent == "wat"):
            temp_solvent0 = torch.ones(1, 14784)
            temp_solvent1 = torch.zeros(1, 14784)
        elif (solvent == "meth"):
            temp_solvent0 = torch.ones(1, 16335)
            temp_solvent1 = torch.zeros(1, 16335)
        for i in range(self.batch_size):
            temp_solvent = temp_solvent0
            for j in range(self.batch_size - i - 1):
                temp_solvent = torch.cat((temp_solvent, temp_solvent1), 1)
            for j in range(i):
                temp_solvent = torch.cat((temp_solvent1, temp_solvent), 1)
            if (i == 0):
                graph_pool_solvent = temp_solvent
            else:
                graph_pool_solvent = torch.cat((graph_pool_solvent, temp_solvent), 0)

        return graph_pool_solute, graph_pool_solvent

    def get_graph_pool_batch(self, len):
        init_batch = torch.LongTensor([0] * len)
        for i in range(self.batch_size):
            if (i == 0):
                continue
            batch = torch.LongTensor([i] * len)
            init_batch = torch.cat((init_batch, batch), 0)
        return init_batch

    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyValModel, self).__init__()

        self.nclass = nclass
        self.batch_size = batch_size
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)

        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(2 * nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.graph_pool_solute_meth, self.graph_pool_solvent_meth = self.get_graph_pool("meth")
        self.DEVICE = DEVICE

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute_adj, solute_meth, solvent_meth, solvent_adj_meth):
        # graph_pool_solute_meth = self.graph_pool_solute_meth.to(self.DEVICE)
        # graph_pool_solvent_meth = self.graph_pool_solvent_meth.to(self.DEVICE)

        init_solute_meth = self.fc1(solute_meth)
        solute_meth = F.relu(self.conv1(solute_meth, solute_adj))
        solute_meth = self.conv2(solute_meth, solute_adj) + init_solute_meth

        # print("solute.shape=",solute.shape)#(batch_size, 2076, 16))
        init_solvent_meth = self.fc1(solvent_meth)
        solvent_meth = F.relu(self.conv1(solvent_meth, solvent_adj_meth))
        solvent_meth = self.conv2(solvent_meth, solvent_adj_meth) + init_solvent_meth
        # print("solvent.shape=", solvent.shape)#(batch_size, 8940, 16))

        len_solute = solute_meth.shape[1]
        len_solvent_meth = solvent_meth.shape[1]
        solute_meth = solute_meth.reshape(-1, self.nclass)#(batch_size*2076,16)
        solvent_meth = solvent_meth.reshape(-1, self.nclass)

        # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        # solute_meth = torch.mm(graph_pool_solute_meth, solute_meth)
        # solvent_meth = torch.mm(graph_pool_solvent_meth, solvent_meth)

        solute_batch = self.get_graph_pool_batch(len_solute).to(self.DEVICE)
        solvent_meth_batch = self.get_graph_pool_batch(len_solvent_meth).to(self.DEVICE)
        solute_meth = gmp(solute_meth, solute_batch)
        solvent_meth = gmp(solvent_meth, solvent_meth_batch)

        data = torch.cat((solute_meth, solvent_meth), 1)
        data = self.dropout(F.relu(self.fc2(data)))
        data = self.dropout(F.relu(self.fc3(data)))
        data = self.dropout(F.relu(self.fc4(data)))
        data = self.fc5(data)

        return data