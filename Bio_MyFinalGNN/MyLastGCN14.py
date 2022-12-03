# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Set2Set
from layers import GraphConvolution
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp

# 新的网络，可以容纳任何溶质
class MyNewGCN(nn.Module):

    # def get_graph_pool(self, solvent):
    #     temp_solute0 = torch.ones(1, 2076)
    #     temp_solute1 = torch.zeros(1, 2076)
    #     for i in range(self.batch_size):
    #         temp_solute = temp_solute0
    #         for j in range(self.batch_size - i - 1):
    #             temp_solute = torch.cat((temp_solute, temp_solute1), 1)
    #         for j in range(i):
    #             temp_solute = torch.cat((temp_solute1, temp_solute), 1)
    #         if(i==0):
    #             graph_pool_solute = temp_solute
    #         else:
    #             graph_pool_solute = torch.cat((graph_pool_solute, temp_solute), 0)
    #     if(solvent == "ACE"):
    #         temp_solvent0 = torch.ones(1, 8940)
    #         temp_solvent1 = torch.zeros(1, 8940)
    #     elif(solvent == "NMF"):
    #         temp_solvent0 = torch.ones(1, 12150)
    #         temp_solvent1 = torch.zeros(1, 12150)
    #     elif (solvent == "wat"):
    #         temp_solvent0 = torch.ones(1, 14784)
    #         temp_solvent1 = torch.zeros(1, 14784)
    #     elif (solvent == "meth"):
    #         temp_solvent0 = torch.ones(1, 16335)
    #         temp_solvent1 = torch.zeros(1, 16335)
    #     for i in range(self.batch_size):
    #         temp_solvent = temp_solvent0
    #         for j in range(self.batch_size - i - 1):
    #             temp_solvent = torch.cat((temp_solvent, temp_solvent1), 1)
    #         for j in range(i):
    #             temp_solvent = torch.cat((temp_solvent1, temp_solvent), 1)
    #         if(i==0):
    #             graph_pool_solvent = temp_solvent
    #         else:
    #             graph_pool_solvent = torch.cat((graph_pool_solvent, temp_solvent), 0)
    #     print("solvent=",solvent,"graph_pool_solute.shape=",graph_pool_solute.shape,"graph_pool_solvent.shape=",graph_pool_solvent.shape)
    #     return graph_pool_solute, graph_pool_solvent

    def get_graph_pool_batch(self, len):
        init_batch = torch.LongTensor([0] * len)
        for i in range(self.batch_size):
            if(i==0):
                continue
            batch = torch.LongTensor([i] * len)
            init_batch = torch.cat((init_batch, batch), 0)
        return init_batch

    def rnn(self, xs):
        print("xs1.shape=", xs.shape)
        xs = torch.unsqueeze(xs, 0)
        print("xs2.shape=", xs.shape)
        xs, h = self.W_rnn(xs)
        print("xs3.shape=", xs.shape)
        xs = torch.relu(xs)
        print("xs4.shape=", xs.shape)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        print("xs5.shape=", xs.shape)
        print("xs6.shape=", torch.unsqueeze(torch.mean(xs, 0), 0).shape)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyNewGCN, self).__init__()

        self.nclass = nclass
        self.batch_size = batch_size
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.solute_conv1 = GCNConv(nfeat, nhid)
        self.solute_conv2 = GCNConv(nhid, nclass)
        
        self.solvent_conv1 = GCNConv(nfeat, nhid)
        self.solvent_conv2 = GCNConv(nhid, nclass)

        smi_embedding_matrix = np.load("./smi_embedding_matrix.npy", allow_pickle=True)
        # self.embed = nn.Embedding(n_fingerprint, dim)
        self.embed_smile = nn.Embedding(100, nclass)
        self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
        self.embed_smile.weight.requires_grad = True

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=nclass)

        self.set2set = Set2Set(nclass, 2, 1)


        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(8 * nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)


        self.dropout = nn.Dropout(p=dropout)
        # self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool("ACE")
        # self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool("NMF")
        # self.graph_pool_solute_meth, self.graph_pool_solvent_meth = self.get_graph_pool("meth")
        # self.graph_pool_solute_wat, self.graph_pool_solvent_wat = self.get_graph_pool("wat")
        self.DEVICE = DEVICE


    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF, solute_wat, solvent_wat, solvent_adj_wat, solute_DMF, solvent_DMF, solvent_adj_DMF, smiles):

        # graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        # graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        # graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        # graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        # graph_pool_solute_wat = self.graph_pool_solute_wat.to(self.DEVICE)
        # graph_pool_solvent_wat = self.graph_pool_solvent_wat.to(self.DEVICE)

        smiles = smiles.to(self.DEVICE)
        solute_smile = smiles[0]
        ACE_solvent_smile = smiles[1]
        NMF_solvent_smile = smiles[2]
        wat_solvent_smile = smiles[3]
        DMF_solvent_smile = smiles[4]

        solute_smile_vectors = self.embed_smile(solute_smile)
        ACE_solvent_smile_vectors = self.embed_smile(ACE_solvent_smile)
        NMF_solvent_smile_vectors = self.embed_smile(NMF_solvent_smile)
        wat_solvent_smile_vectors = self.embed_smile(wat_solvent_smile)
        DMF_solvent_smile_vectors = self.embed_smile(DMF_solvent_smile)

        after_solute_smile_vectors = self.rnn(solute_smile_vectors).repeat(self.batch_size, 1)
        after_ACE_solvent_smile_vectors = self.rnn(ACE_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_NMF_solvent_smile_vectors = self.rnn(NMF_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_wat_solvent_smile_vectors = self.rnn(wat_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_DMF_solvent_smile_vectors = self.rnn(DMF_solvent_smile_vectors).repeat(self.batch_size, 1)


        init_solute_ACE = self.fc1(solute_ACE)
        init_solute_NMF = self.fc1(solute_NMF)
        init_solute_wat = self.fc1(solute_wat)
        init_solute_DMF = self.fc1(solute_DMF)

        #溶质进行3层GCN+resnet
        solute_ACE = F.relu(self.solute_conv1(solute_ACE, solute_adj))
        solute_ACE = self.solute_conv2(solute_ACE, solute_adj)+ init_solute_ACE
        solute_NMF = F.relu(self.solute_conv1(solute_NMF, solute_adj))
        solute_NMF = self.solute_conv2(solute_NMF, solute_adj) + init_solute_NMF
        solute_wat = F.relu(self.solute_conv1(solute_wat, solute_adj))
        solute_wat = self.solute_conv2(solute_wat, solute_adj) + init_solute_wat
        solute_DMF = F.relu(self.solute_conv1(solute_DMF, solute_adj))
        solute_DMF = self.solute_conv2(solute_DMF, solute_adj) + init_solute_DMF
        # print("solute.shape=",solute.shape)#(batch_size, 2076, 16))

        init_solvent_ACE = self.fc1(solvent_ACE)
        init_solvent_NMF = self.fc1(solvent_NMF)
        init_solvent_wat = self.fc1(solvent_wat)
        init_solvent_DMF = self.fc1(solvent_DMF)
        # 溶剂进行3层GCN+resnet
        solvent_ACE = F.relu(self.solvent_conv1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.solvent_conv2(solvent_ACE, solvent_adj_ACE) + init_solvent_ACE
        solvent_NMF = F.relu(self.solvent_conv1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.solvent_conv2(solvent_NMF, solvent_adj_NMF) + init_solvent_NMF
        solvent_wat = F.relu(self.solvent_conv1(solvent_wat, solvent_adj_wat))
        solvent_wat = self.solvent_conv2(solvent_wat, solvent_adj_wat) + init_solvent_wat
        solvent_DMF = F.relu(self.solvent_conv1(solvent_DMF, solvent_adj_DMF))
        solvent_DMF = self.solvent_conv2(solvent_DMF, solvent_adj_DMF) + init_solvent_DMF
        # print("solvent.shape=", solvent.shape)#(batch_size, 8940, 16))
        # print("solvent_wat.shape=", solvent_wat.shape)
        # print("solute_ACE.shape=", solute_ACE.shape)

        len_solute = solute_ACE.shape[1]
        len_solvent_ACE = solvent_ACE.shape[1]
        len_solvent_NMF = solvent_NMF.shape[1]
        len_solvent_wat = solvent_wat.shape[1]
        len_solvent_DMF = solvent_DMF.shape[1]

        solute_ACE = solute_ACE.reshape(-1, self.nclass)
        solvent_ACE = solvent_ACE.reshape(-1, self.nclass)
        solute_NMF = solute_NMF.reshape(-1, self.nclass)
        solvent_NMF = solvent_NMF.reshape(-1, self.nclass)
        solute_wat = solute_wat.reshape(-1, self.nclass)
        solvent_wat = solvent_wat.reshape(-1, self.nclass)
        solute_DMF = solute_DMF.reshape(-1, self.nclass)
        solvent_DMF = solvent_DMF.reshape(-1, self.nclass)
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
        solvent_DMF_batch = self.get_graph_pool_batch(len_solvent_DMF).to(self.DEVICE)
        solute_ACE = self.set2set(solute_ACE, solute_batch)
        solute_NMF = self.set2set(solute_NMF, solute_batch)
        solute_wat = self.set2set(solute_wat, solute_batch)
        solute_DMF = self.set2set(solute_DMF, solute_batch)
        solvent_ACE = self.set2set(solvent_ACE, solvent_ACE_batch)
        solvent_NMF = self.set2set(solvent_NMF, solvent_NMF_batch)
        solvent_wat = self.set2set(solvent_wat, solvent_wat_batch)
        solvent_DMF = self.set2set(solvent_DMF, solvent_DMF_batch)
        
        data0 = torch.cat((solute_ACE, after_solute_smile_vectors), 1)
        data1 = torch.cat((solvent_ACE, after_ACE_solvent_smile_vectors), 1)
        data1 = torch.cat((data0, data1), 1)
        data1 = self.dropout(F.relu(self.fc2(data1)))
        data1 = self.dropout(F.relu(self.fc3(data1)))
        data1 = self.dropout(F.relu(self.fc4(data1)))
        data1 = self.fc5(data1)

        data0 = torch.cat((solute_NMF, after_solute_smile_vectors), 1)
        data2 = torch.cat((solvent_NMF, after_NMF_solvent_smile_vectors), 1)
        data2 = torch.cat((data0, data2), 1)
        data2 = self.dropout(F.relu(self.fc2(data2)))
        data2 = self.dropout(F.relu(self.fc3(data2)))
        data2 = self.dropout(F.relu(self.fc4(data2)))
        data2 = self.fc5(data2)

        data0 = torch.cat((solute_wat, after_solute_smile_vectors), 1)
        data3 = torch.cat((solvent_wat, after_wat_solvent_smile_vectors), 1)
        data3 = torch.cat((data0, data3), 1)
        data3 = self.dropout(F.relu(self.fc2(data3)))
        data3 = self.dropout(F.relu(self.fc3(data3)))
        data3 = self.dropout(F.relu(self.fc4(data3)))
        data3 = self.fc5(data3)
        
        data0 = torch.cat((solute_DMF, after_solute_smile_vectors), 1)
        data4 = torch.cat((solvent_DMF, after_DMF_solvent_smile_vectors), 1)
        data4 = torch.cat((data0, data4), 1)
        data4 = self.dropout(F.relu(self.fc2(data4)))
        data4 = self.dropout(F.relu(self.fc3(data4)))
        data4 = self.dropout(F.relu(self.fc4(data4)))
        data4 = self.fc5(data4)

        data = torch.cat((data1, data2), 0)
        data = torch.cat((data, data3), 0)
        data = torch.cat((data, data4), 0)

        return data