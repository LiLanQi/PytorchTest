# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import GraphConvolution
from torch.nn.parameter import Parameter

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
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE):
        super(MyNewGCN, self).__init__()

        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(2 * nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.fc0 = nn.Linear(nclass, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool_ACE()
        self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool_NMF()
        self.graph_pool_solute_meth, self.graph_pool_solvent_meth = self.get_graph_pool_meth()
        self.graph_pool_solute_wat, self.graph_pool_solvent_wat = self.get_graph_pool_wat()
        self.DEVICE = DEVICE



    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF, solute_wat, solvent_wat, solvent_adj_wat):

        graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        graph_pool_solute_wat = self.graph_pool_solute_wat.to(self.DEVICE)
        graph_pool_solvent_wat = self.graph_pool_solvent_wat.to(self.DEVICE)

        # len_map_ACE = torch.mm(graph_pool_solute_ACE.t(), graph_pool_solvent_ACE)
        # len_map_NMF = torch.mm(graph_pool_solute_NMF.t(), graph_pool_solvent_NMF)
        # len_map_ACE = torch.mm(graph_pool_solute_wat.t(), graph_pool_solvent_wat)
        # print("len_map_ACE.shape=",len_map_ACE.shape)

        init_solute_ACE = self.fc1(solute_ACE)
        init_solute_NMF = self.fc1(solute_NMF)
        init_solute_wat = self.fc1(solute_wat)

        solute_ACE = F.relu(self.gc1(solute_ACE, solute_adj))
        solute_ACE = self.gc2(solute_ACE, solute_adj) + init_solute_ACE
        solute_NMF = F.relu(self.gc1(solute_NMF, solute_adj))
        solute_NMF = self.gc2(solute_NMF, solute_adj) + init_solute_NMF
        solute_wat = F.relu(self.gc1(solute_wat, solute_adj))
        solute_wat = self.gc2(solute_wat, solute_adj) + init_solute_wat

        # print("solute.shape=",solute.shape)#(4, 2076, 16))
        init_solvent_ACE = self.fc1(solvent_ACE)
        init_solvent_NMF = self.fc1(solvent_NMF)
        init_solvent_wat = self.fc1(solvent_wat)
        solvent_ACE = F.relu(self.gc1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.gc2(solvent_ACE, solvent_adj_ACE) + init_solvent_ACE
        solvent_NMF = F.relu(self.gc1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.gc2(solvent_NMF, solvent_adj_NMF) + init_solvent_NMF
        solvent_wat = F.relu(self.gc1(solvent_wat, solvent_adj_wat))
        solvent_wat = self.gc2(solvent_wat, solvent_adj_wat) + init_solvent_wat
        # print("solvent.shape=", solvent.shape)#(4, 8940, 16))

        solute_ACE = solute_ACE.reshape(-1, self.nclass)
        solvent_ACE = solvent_ACE.reshape(-1, self.nclass)
        solute_NMF = solute_NMF.reshape(-1, self.nclass)
        solvent_NMF = solvent_NMF.reshape(-1, self.nclass)
        solute_wat = solute_wat.reshape(-1, self.nclass)
        solvent_wat = solvent_wat.reshape(-1, self.nclass)
        solute_ACE = self.fc0(solute_ACE)
        print("solute_ACE.shape=",solute_ACE.shape)
        solute_and_ACE_interaction = torch.tanh(torch.mm(solute_ACE, solvent_ACE.t()))
        print("solute_and_ACE_interaction.shape=", solute_and_ACE_interaction.shape)
        solute_and_NMF_interaction = torch.tanh(torch.mm(self.fc0(solute_NMF), solvent_NMF.t()))
        print("solute_and_NMF_interaction.shape=", solute_and_NMF_interaction.shape)
        solute_and_wat_interaction = torch.tanh(torch.mm(self.fc0(solute_wat), solvent_wat.t()))
        print("solute_and_wat_interaction.shape=", solute_and_wat_interaction.shape)

        A_solute_and_ACE_interaction = solute_and_ACE_interaction

        # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)
        solute_wat = torch.mm(graph_pool_solute_wat, solute_wat)
        solvent_wat = torch.mm(graph_pool_solvent_wat, solvent_wat)



        data1 = torch.cat((solute_ACE, solvent_ACE), 1)
        data1 = self.dropout(F.relu(self.fc2(data1)))
        data1 = self.dropout(F.relu(self.fc3(data1)))
        data1 = self.dropout(F.relu(self.fc4(data1)))
        data1 = self.fc5(data1)

        data2 = torch.cat((solute_NMF, solvent_NMF), 1)
        data2 = self.dropout(F.relu(self.fc2(data2)))
        data2 = self.dropout(F.relu(self.fc3(data2)))
        data2 = self.dropout(F.relu(self.fc4(data2)))
        data2 = self.fc5(data2)

        data3 = torch.cat((solute_wat, solvent_wat), 1)
        data3 = self.dropout(F.relu(self.fc2(data3)))
        data3 = self.dropout(F.relu(self.fc3(data3)))
        data3 = self.dropout(F.relu(self.fc4(data3)))
        data3 = self.fc5(data3)

        data = torch.cat((data1, data2), 0)
        data = torch.cat((data, data3), 0)

        return data