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

    def get_graph_pool_wat(self):
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

        temp_solvent0 = torch.ones(1, 14784)
        temp_solvent1 = torch.zeros(1, 14784)
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

    def get_graph_pool_meth(self):

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

        temp_solvent0 = torch.ones(1, 16335)
        temp_solvent1 = torch.zeros(1, 16335)
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
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyNewGCN, self).__init__()

        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(2*nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)


        self.dropout = dropout
        self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool_ACE()
        self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool_NMF()
        self.graph_pool_solute_meth, self.graph_pool_solvent_meth = self.get_graph_pool_meth()
        self.DEVICE = torch.device('cuda:1')

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF, solute_meth, solvent_meth, solvent_adj_meth):

        graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        graph_pool_solute_meth = self.graph_pool_solute_meth.to(self.DEVICE)
        graph_pool_solvent_meth = self.graph_pool_solvent_meth.to(self.DEVICE)

        init_solute_ACE = self.fc1(solute_ACE)
        init_solute_NMF = self.fc1(solute_NMF)
        init_solute_meth= self.fc1(solute_meth)

        solute_ACE = F.relu(self.gc1(solute_ACE, solute_adj))
        solute_ACE = self.gc2(solute_ACE, solute_adj) + init_solute_ACE
        solute_NMF = F.relu(self.gc1(solute_NMF, solute_adj))
        solute_NMF = self.gc2(solute_NMF, solute_adj) + init_solute_NMF
        solute_meth = F.relu(self.gc1(solute_meth, solute_adj))
        solute_meth = self.gc2(solute_meth, solute_adj) + init_solute_meth

        # print("solute.shape=",solute.shape)#(4, 2076, 16))
        init_solvent_ACE = self.fc1(solvent_ACE)
        init_solvent_NMF = self.fc1(solvent_NMF)
        init_solvent_meth = self.fc1(solvent_meth)
        solvent_ACE = F.relu(self.gc1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.gc2(solvent_ACE, solvent_adj_ACE) + init_solvent_ACE
        solvent_NMF = F.relu(self.gc1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.gc2(solvent_NMF, solvent_adj_NMF) + init_solvent_NMF
        solvent_meth = F.relu(self.gc1(solvent_meth, solvent_adj_meth))
        solvent_meth = self.gc2(solvent_meth, solvent_adj_meth) + init_solvent_meth
        # print("solvent.shape=", solvent.shape)#(4, 8940, 16))

        solute_ACE = solute_ACE.reshape(-1, self.nclass)
        solvent_ACE = solvent_ACE.reshape(-1, self.nclass)
        solute_NMF = solute_NMF.reshape(-1, self.nclass)
        solvent_NMF = solvent_NMF.reshape(-1, self.nclass)
        solute_meth = solute_meth.reshape(-1, self.nclass)
        solvent_meth = solvent_meth.reshape(-1, self.nclass)

        # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)
        solute_meth = torch.mm(graph_pool_solute_meth, solute_meth)
        solvent_meth = torch.mm(graph_pool_solvent_meth, solvent_meth)

        data1 = torch.cat((solute_ACE, solvent_ACE), 1)
        data1 = F.relu(self.fc2(data1))
        data1 = F.relu(self.fc3(data1))
        data1 = F.relu(self.fc4(data1))
        data1 = self.fc5(data1)

        data2 = torch.cat((solute_NMF, solvent_NMF), 1)
        data2 = F.relu(self.fc2(data2))
        data2 = F.relu(self.fc3(data2))
        data2 = F.relu(self.fc4(data2))
        data2 = self.fc5(data2)

        data3 = torch.cat((solute_meth, solvent_meth), 1)
        data3 = F.relu(self.fc2(data3))
        data3 = F.relu(self.fc3(data3))
        data3 = F.relu(self.fc4(data3))
        data3 = self.fc5(data3)

        data = torch.cat((data1, data2), 0)
        data = torch.cat((data, data3), 0)
        return data