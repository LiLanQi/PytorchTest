# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Set2Set
from layers import GraphConvolution
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp

# 新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):

    def get_graph_pool(self, solvent):
        temp_solute0 = torch.ones(1, 2076)
        temp_solute1 = torch.zeros(1, 2076)
        for i in range(self.batch_size):
            temp_solute = temp_solute0
            for j in range(self.batch_size - i - 1):
                temp_solute = torch.cat((temp_solute, temp_solute1), 1)
            for j in range(i):
                temp_solute = torch.cat((temp_solute1, temp_solute), 1)
            if(i==0):
                graph_pool_solute = temp_solute
            else:
                graph_pool_solute = torch.cat((graph_pool_solute, temp_solute), 0)
        if(solvent == "ACE"):
            temp_solvent0 = torch.ones(1, 8940)
            temp_solvent1 = torch.zeros(1, 8940)
        elif(solvent == "NMF"):
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
            if(i==0):
                graph_pool_solvent = temp_solvent
            else:
                graph_pool_solvent = torch.cat((graph_pool_solvent, temp_solvent), 0)
        print("solvent=",solvent,"graph_pool_solute.shape=",graph_pool_solute.shape,"graph_pool_solvent.shape=",graph_pool_solvent.shape)
        return graph_pool_solute, graph_pool_solvent

    def get_graph_pool_batch(self, len):
        init_batch = torch.LongTensor([0] * len)
        for i in range(self.batch_size):
            if(i==0):
                continue
            batch = torch.LongTensor([i] * len)
            init_batch = torch.cat((init_batch, batch), 0)
        return init_batch

    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyNewGCN, self).__init__()

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
        self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool("ACE")
        self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool("NMF")
        self.graph_pool_solute_meth, self.graph_pool_solvent_meth = self.get_graph_pool("meth")
        self.graph_pool_solute_wat, self.graph_pool_solvent_wat = self.get_graph_pool("wat")
        self.DEVICE = DEVICE


    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF, solute_wat, solvent_wat, solvent_adj_wat):

        # graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        # graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        # graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        # graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        # graph_pool_solute_wat = self.graph_pool_solute_wat.to(self.DEVICE)
        # graph_pool_solvent_wat = self.graph_pool_solvent_wat.to(self.DEVICE)
        init_solute_ACE = self.fc1(solute_ACE)
        init_solute_NMF = self.fc1(solute_NMF)
        init_solute_wat = self.fc1(solute_wat)

        #溶质进行3层GCN+resnet
        solute_ACE = F.relu(self.conv1(solute_ACE, solute_adj))
        solute_ACE = self.conv2(solute_ACE, solute_adj)+ init_solute_ACE
        solute_NMF = self.conv1(solute_NMF, solute_adj)
        solute_NMF = self.conv2(solute_NMF, solute_adj) + init_solute_NMF
        solute_wat = F.relu(self.conv1(solute_wat, solute_adj))
        solute_wat = self.conv2(solute_wat, solute_adj) + init_solute_wat
        # print("solute.shape=",solute.shape)#(batch_size, 2076, 16))

        init_solvent_ACE = self.fc1(solvent_ACE)
        init_solvent_NMF = self.fc1(solvent_NMF)
        init_solvent_wat = self.fc1(solvent_wat)
        # 溶剂进行3层GCN+resnet
        solvent_ACE = F.relu(self.conv1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.conv2(solvent_ACE, solvent_adj_ACE) + init_solvent_ACE
        solvent_NMF = F.relu(self.conv1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.conv2(solvent_NMF, solvent_adj_NMF) + init_solvent_NMF
        solvent_wat = F.relu(self.conv1(solvent_wat, solvent_adj_wat))
        solvent_wat = self.conv2(solvent_wat, solvent_adj_wat) + init_solvent_wat
        # print("solvent.shape=", solvent.shape)#(batch_size, 8940, 16))
        # print("solvent_wat.shape=", solvent_wat.shape)
        # print("solute_ACE.shape=", solute_ACE.shape)

        len_solute = solute_ACE.shape[1]
        len_solvent_ACE = solvent_ACE.shape[1]
        len_solvent_NMF = solvent_NMF.shape[1]
        len_solvent_wat = solvent_wat.shape[1]

        solute_ACE = solute_ACE.reshape(-1, self.nclass)
        solvent_ACE = solvent_ACE.reshape(-1, self.nclass)
        solute_NMF = solute_NMF.reshape(-1, self.nclass)
        solvent_NMF = solvent_NMF.reshape(-1, self.nclass)
        solute_wat = solute_wat.reshape(-1, self.nclass)
        solvent_wat = solvent_wat.reshape(-1, self.nclass)
        #
        # # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        # solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        # solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        # solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        # solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)
        # solute_wat = torch.mm(graph_pool_solute_wat, solute_wat)
        # solvent_wat = torch.mm(graph_pool_solvent_wat, solvent_wat)
        solute_batch = self.get_graph_pool_batch(len_solute).to(self.DEVICE)
        solvent_ACE_batch = self.get_graph_pool_batch(len_solvent_ACE).to(self.DEVICE)
        solvent_NMF_batch = self.get_graph_pool_batch(len_solvent_NMF).to(self.DEVICE)
        solvent_wat_batch = self.get_graph_pool_batch(len_solvent_wat).to(self.DEVICE)
        solute_ACE = gmp(solute_ACE, solute_batch)
        solute_NMF = gmp(solute_NMF, solute_batch)
        solute_wat = gmp(solute_wat, solute_batch)
        solvent_ACE = gmp(solvent_ACE, solvent_ACE_batch)
        solvent_NMF = gmp(solvent_NMF, solvent_NMF_batch)
        solvent_wat = gmp(solvent_wat, solvent_wat_batch)

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