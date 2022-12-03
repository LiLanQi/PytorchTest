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



    def get_graph_pool_batch(self, len):
        init_batch = torch.LongTensor([0] * len)
        for i in range(self.batch_size):
            if (i == 0):
                continue
            batch = torch.LongTensor([i] * len)
            init_batch = torch.cat((init_batch, batch), 0)
        return init_batch

    def rnn(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyNewGCN, self).__init__()

        self.nclass = nclass
        self.batch_size = batch_size
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)

        self.solute_conv1 = SAGEConv(nfeat, nhid)
        self.solute_conv2 = SAGEConv(nhid, nclass)

        self.solvent_conv1 = SAGEConv(nfeat, nhid)
        self.solvent_conv2 = SAGEConv(nhid, nclass)

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

    def forward(self, solute_meth, solvent_meth, solute_adj, solvent_adj_meth, solute_NMF, solvent_NMF, solvent_adj_NMF,
                solute_wat, solvent_wat, solvent_adj_wat, solute_DMF, solvent_DMF, solvent_adj_DMF, smiles):

        # graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        # graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        # graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        # graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        # graph_pool_solute_wat = self.graph_pool_solute_wat.to(self.DEVICE)
        # graph_pool_solvent_wat = self.graph_pool_solvent_wat.to(self.DEVICE)

        smiles = smiles.to(self.DEVICE)
        solute_smile = smiles[0]
        meth_solvent_smile = smiles[5] #其实是ACE
        NMF_solvent_smile = smiles[2]
        wat_solvent_smile = smiles[3]
        DMF_solvent_smile = smiles[4]

        solute_smile_vectors = self.embed_smile(solute_smile)
        meth_solvent_smile_vectors = self.embed_smile(meth_solvent_smile)
        NMF_solvent_smile_vectors = self.embed_smile(NMF_solvent_smile)
        wat_solvent_smile_vectors = self.embed_smile(wat_solvent_smile)
        DMF_solvent_smile_vectors = self.embed_smile(DMF_solvent_smile)

        after_solute_smile_vectors = self.rnn(solute_smile_vectors).repeat(self.batch_size, 1)
        after_meth_solvent_smile_vectors = self.rnn(meth_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_NMF_solvent_smile_vectors = self.rnn(NMF_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_wat_solvent_smile_vectors = self.rnn(wat_solvent_smile_vectors).repeat(self.batch_size, 1)
        after_DMF_solvent_smile_vectors = self.rnn(DMF_solvent_smile_vectors).repeat(self.batch_size, 1)

        init_solute_meth = self.fc1(solute_meth)
        init_solute_NMF = self.fc1(solute_NMF)
        init_solute_wat = self.fc1(solute_wat)
        init_solute_DMF = self.fc1(solute_DMF)

        # 溶质进行3层GCN+resnet
        solute_meth = F.relu(self.solute_conv1(solute_meth, solute_adj))
        solute_meth = self.solute_conv2(solute_meth, solute_adj) + init_solute_meth
        solute_NMF = F.relu(self.solute_conv1(solute_NMF, solute_adj))
        solute_NMF = self.solute_conv2(solute_NMF, solute_adj) + init_solute_NMF
        solute_wat = F.relu(self.solute_conv1(solute_wat, solute_adj))
        solute_wat = self.solute_conv2(solute_wat, solute_adj) + init_solute_wat
        solute_DMF = F.relu(self.solute_conv1(solute_DMF, solute_adj))
        solute_DMF = self.solute_conv2(solute_DMF, solute_adj) + init_solute_DMF
        # print("solute.shape=",solute.shape)#(batch_size, 2076, 16))

        init_solvent_meth = self.fc1(solvent_meth)
        init_solvent_NMF = self.fc1(solvent_NMF)
        init_solvent_wat = self.fc1(solvent_wat)
        init_solvent_DMF = self.fc1(solvent_DMF)
        # 溶剂进行3层GCN+resnet
        solvent_meth = F.relu(self.solvent_conv1(solvent_meth, solvent_adj_meth))
        solvent_meth = self.solvent_conv2(solvent_meth, solvent_adj_meth) + init_solvent_meth
        solvent_NMF = F.relu(self.solvent_conv1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.solvent_conv2(solvent_NMF, solvent_adj_NMF) + init_solvent_NMF
        solvent_wat = F.relu(self.solvent_conv1(solvent_wat, solvent_adj_wat))
        solvent_wat = self.solvent_conv2(solvent_wat, solvent_adj_wat) + init_solvent_wat
        solvent_DMF = F.relu(self.solvent_conv1(solvent_DMF, solvent_adj_DMF))
        solvent_DMF = self.solvent_conv2(solvent_DMF, solvent_adj_DMF) + init_solvent_DMF
        # print("solvent.shape=", solvent.shape)#(batch_size, 8940, 16))
        # print("solvent_wat.shape=", solvent_wat.shape)
        # print("solute_ACE.shape=", solute_ACE.shape)

        len_solute = solute_meth.shape[1]
        len_solvent_meth = solvent_meth.shape[1]
        len_solvent_NMF = solvent_NMF.shape[1]
        len_solvent_wat = solvent_wat.shape[1]
        len_solvent_DMF = solvent_DMF.shape[1]

        solute_meth = solute_meth.reshape(-1, self.nclass)
        solvent_meth = solvent_meth.reshape(-1, self.nclass)
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
        solvent_meth_batch = self.get_graph_pool_batch(len_solvent_meth).to(self.DEVICE)
        solvent_NMF_batch = self.get_graph_pool_batch(len_solvent_NMF).to(self.DEVICE)
        solvent_wat_batch = self.get_graph_pool_batch(len_solvent_wat).to(self.DEVICE)
        solvent_DMF_batch = self.get_graph_pool_batch(len_solvent_DMF).to(self.DEVICE)
        solute_meth = self.set2set(solute_meth, solute_batch)
        solute_NMF = self.set2set(solute_NMF, solute_batch)
        solute_wat = self.set2set(solute_wat, solute_batch)
        solute_DMF = self.set2set(solute_DMF, solute_batch)
        solvent_meth = self.set2set(solvent_meth, solvent_meth_batch)
        solvent_NMF = self.set2set(solvent_NMF, solvent_NMF_batch)
        solvent_wat = self.set2set(solvent_wat, solvent_wat_batch)
        solvent_DMF = self.set2set(solvent_DMF, solvent_DMF_batch)

        data0 = torch.cat((solute_meth, after_solute_smile_vectors), 1)
        data1 = torch.cat((solvent_meth, after_meth_solvent_smile_vectors), 1)
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