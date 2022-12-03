# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, Set2Set, GINConv
from layers import GraphConvolution
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gdp


#
# class Attention(nn.Module):
#     """
#     Obtained from: github.com:rwightman/pytorch-image-models
#     """
#
#     def __init__(self, dim, num_heads=1, attention_dropout=0.1, projection_dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.attn_drop = nn.Dropout(attention_dropout)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         return x
#

# 新的网络，可以容纳任何溶质
class MyNewGNN(nn.Module):

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
        xs = torch.squeeze(xs, 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyNewGNN, self).__init__()

        self.nclass = nclass
        self.batch_size = batch_size

        self.solute_conv1 = SAGEConv(nclass, nclass)
        self.solute_conv2 = SAGEConv(nclass, nclass)

        self.solvent_conv1 = SAGEConv(nclass, nclass)
        self.solvent_conv2 = SAGEConv(nclass, nclass)

        # smi_embedding_matrix = np.load("./smi_embedding_matrix.npy", allow_pickle=True)
        # self.embed = nn.Embedding(n_fingerprint, dim)
        # self.embed_smile = nn.Embedding(100, nclass)
        # self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
        # self.embed_smile.weight.requires_grad = True

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=300, hidden_size=nclass)

        self.set2set = Set2Set(nclass, 2, 1)

        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(8 * nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=dropout)

        self.DEVICE = DEVICE

    def forward(self, solute_data_zero, solvent_data_zero, solute, zero_solvent,
                solute_data_one, solvent_data_one, one_solvent,
                solute_data_two, solvent_data_two, two_solvent,
                solute_data_three, solvent_data_three, three_solvent):

        len_solute = solute_data_zero.shape[1]
        len_solvent_zero = solvent_data_zero.shape[1]
        len_solvent_one = solvent_data_one.shape[1]
        len_solvent_two = solvent_data_two.shape[1]
        len_solvent_three = solvent_data_three.shape[1]
        after_solute_smile_vectors = self.rnn(solute["solute_to_embedding"]).repeat(self.batch_size, 1)
        after_zero_solvent_smile_vectors = self.rnn(zero_solvent["smile_to_vector"]).repeat(self.batch_size, 1)
        after_one_solvent_smile_vectors = self.rnn(one_solvent["smile_to_vector"]).repeat(self.batch_size, 1)
        after_two_solvent_smile_vectors = self.rnn(two_solvent["smile_to_vector"]).repeat(self.batch_size, 1)
        after_three_solvent_smile_vectors = self.rnn(three_solvent["smile_to_vector"]).repeat(self.batch_size, 1)
        solute_data_zero = solute_data_zero.reshape(-1, solute_data_zero.shape[-1])
        solute_data_one = solute_data_one.reshape(-1, solute_data_one.shape[-1])
        solute_data_two = solute_data_two.reshape(-1, solute_data_two.shape[-1])
        solute_data_three = solute_data_three.reshape(-1, solute_data_three.shape[-1])
        init_solute_zero = self.fc1(solute_data_zero)
        # print("solute_data_zero.shape=", solute_data_zero.shape) #solute_data_zero.shape= torch.Size([4, 2076, 9])
        # print("init_solute_zero.shape=", init_solute_zero.shape) #init_solute_zero.shape= torch.Size([4, 2076, 9, 128])
        init_solute_one = self.fc1(solute_data_one)
        init_solute_two = self.fc1(solute_data_two)
        init_solute_three = self.fc1(solute_data_three)

        # 溶质进行2层GraphSAGE+resnet

        solute_data_zero = F.relu(self.solute_conv1(init_solute_zero, solute["solute_adj"]))
        solute_data_zero = self.solute_conv2(solute_data_zero, solute["solute_adj"]) + init_solute_zero
        solute_data_one = F.relu(self.solute_conv1(init_solute_one, solute["solute_adj"]))
        solute_data_one = self.solute_conv2(solute_data_one, solute["solute_adj"]) + init_solute_one
        solute_data_two = F.relu(self.solute_conv1(init_solute_two, solute["solute_adj"]))
        solute_data_two = self.solute_conv2(solute_data_two, solute["solute_adj"]) + init_solute_two
        solute_data_three = F.relu(self.solute_conv1(init_solute_three, solute["solute_adj"]))
        solute_data_three = self.solute_conv2(solute_data_three, solute["solute_adj"]) + init_solute_three
        # print("solute.shape=",solute.shape)#(batch_size, 2076, 16))
        solvent_data_zero = solvent_data_zero.reshape(-1, solvent_data_zero.shape[-1])
        solvent_data_one = solvent_data_one.reshape(-1, solvent_data_one.shape[-1])
        solvent_data_two = solvent_data_two.reshape(-1, solvent_data_two.shape[-1])
        solvent_data_three = solvent_data_three.reshape(-1, solvent_data_three.shape[-1])

        init_solvent_zero = self.fc1(solvent_data_zero)
        init_solvent_one = self.fc1(solvent_data_one)
        init_solvent_two = self.fc1(solvent_data_two)
        init_solvent_three = self.fc1(solvent_data_three)

        # 溶剂进行2层GraphSAGE+resnet
        solvent_data_zero = F.relu(self.solvent_conv1(init_solvent_zero, zero_solvent["solvent_adj"]))

        solvent_data_zero = self.solvent_conv2(solvent_data_zero, zero_solvent["solvent_adj"]) + init_solvent_zero
        solvent_data_one = F.relu(self.solvent_conv1(init_solvent_one, one_solvent["solvent_adj"]))
        solvent_data_one = self.solvent_conv2(solvent_data_one, one_solvent["solvent_adj"]) + init_solvent_one
        solvent_data_two = F.relu(self.solvent_conv1(init_solvent_two, two_solvent["solvent_adj"]))
        solvent_data_two = self.solvent_conv2(solvent_data_two, two_solvent["solvent_adj"]) + init_solvent_two
        solvent_data_three = F.relu(self.solvent_conv1(init_solvent_three, three_solvent["solvent_adj"]))
        solvent_data_three = self.solvent_conv2(solvent_data_three, three_solvent["solvent_adj"]) + init_solvent_three

        # # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        # solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        # solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        # solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        # solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)
        # solute_wat = torch.mm(graph_pool_solute_wat, solute_wat)
        # solvent_wat = torch.mm(graph_pool_solvent_wat, solvent_wat)
        solute_batch = self.get_graph_pool_batch(len_solute).to(self.DEVICE)
        solvent_zero_batch = self.get_graph_pool_batch(len_solvent_zero).to(self.DEVICE)
        solvent_one_batch = self.get_graph_pool_batch(len_solvent_one).to(self.DEVICE)
        solvent_two_batch = self.get_graph_pool_batch(len_solvent_two).to(self.DEVICE)
        solvent_three_batch = self.get_graph_pool_batch(len_solvent_three).to(self.DEVICE)
        solute_zero = self.set2set(solute_data_zero, solute_batch)
        solute_one = self.set2set(solute_data_one, solute_batch)
        solute_two = self.set2set(solute_data_two, solute_batch)
        solute_three = self.set2set(solute_data_three, solute_batch)
        solvent_zero = self.set2set(solvent_data_zero, solvent_zero_batch)
        solvent_one = self.set2set(solvent_data_one, solvent_one_batch)
        solvent_two = self.set2set(solvent_data_two, solvent_two_batch)
        solvent_three = self.set2set(solvent_data_three, solvent_three_batch)

        # print("solute_zero.shape=", solute_zero.shape) #(batch_size,nclass)
        # solute_solvent_zero = torch.cat((solute_zero, solvent_zero), )

        # n_solvent = solvent_zero.size(0)

        data0 = torch.cat((solute_zero, after_solute_smile_vectors), 1)
        data1 = torch.cat((solvent_zero, after_zero_solvent_smile_vectors), 1)
        data1 = torch.cat((data0, data1), 1)
        data1 = self.dropout(F.relu(self.fc2(data1)))
        data1 = self.dropout(F.relu(self.fc3(data1)))
        data1 = self.dropout(F.relu(self.fc4(data1)))
        data1 = self.fc5(data1)

        data0 = torch.cat((solute_one, after_solute_smile_vectors), 1)
        data2 = torch.cat((solvent_one, after_one_solvent_smile_vectors), 1)
        data2 = torch.cat((data0, data2), 1)
        data2 = self.dropout(F.relu(self.fc2(data2)))
        data2 = self.dropout(F.relu(self.fc3(data2)))
        data2 = self.dropout(F.relu(self.fc4(data2)))
        data2 = self.fc5(data2)

        data0 = torch.cat((solute_two, after_solute_smile_vectors), 1)
        data3 = torch.cat((solvent_two, after_two_solvent_smile_vectors), 1)
        data3 = torch.cat((data0, data3), 1)
        data3 = self.dropout(F.relu(self.fc2(data3)))
        data3 = self.dropout(F.relu(self.fc3(data3)))
        data3 = self.dropout(F.relu(self.fc4(data3)))
        data3 = self.fc5(data3)

        data0 = torch.cat((solute_three, after_solute_smile_vectors), 1)
        data4 = torch.cat((solvent_three, after_three_solvent_smile_vectors), 1)
        data4 = torch.cat((data0, data4), 1)
        data4 = self.dropout(F.relu(self.fc2(data4)))
        data4 = self.dropout(F.relu(self.fc3(data4)))
        data4 = self.dropout(F.relu(self.fc4(data4)))
        data4 = self.fc5(data4)
        data = torch.cat((data1, data2), 0)
        data = torch.cat((data, data3), 0)
        data = torch.cat((data, data4), 0)

        return data


class MyValModel(nn.Module):

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
        xs = torch.squeeze(xs, 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def __init__(self, nfeat, nhid, nclass, dropout, DEVICE, batch_size):
        super(MyValModel, self).__init__()

        self.nclass = nclass
        self.batch_size = batch_size

        self.solute_conv1 = SAGEConv(nclass, nclass)
        self.solute_conv2 = SAGEConv(nclass, nclass)

        self.solvent_conv1 = SAGEConv(nclass, nclass)
        self.solvent_conv2 = SAGEConv(nclass, nclass)

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=300, hidden_size=nclass)

        self.set2set = Set2Set(nclass, 2, 1)

        self.fc1 = nn.Linear(nfeat, nclass)

        self.fc2 = nn.Linear(8 * nclass, nclass)
        self.fc3 = nn.Linear(nclass, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=dropout)

        self.DEVICE = DEVICE

    def forward(self, solute, solute_data_test, solvent_data_test, solvent_test):

        len_solute = solute_data_test.shape[1]
        len_solvent_test = solvent_data_test.shape[1]

        after_solute_smile_vectors = self.rnn(solute["solute_to_embedding"]).repeat(self.batch_size, 1)
        after_test_solvent_smile_vectors = self.rnn(solvent_test["smile_to_vector"]).repeat(self.batch_size, 1)

        solute_data_test = solute_data_test.reshape(-1, solute_data_test.shape[-1])
        init_solute_test = self.fc1(solute_data_test)
        # 溶质进行2层GraphSAGE+resnet
        solute_data_test = F.relu(self.solute_conv1(init_solute_test, solute["solute_adj"]))
        solute_data_test = self.solute_conv2(solute_data_test, solute["solute_adj"]) + init_solute_test

        solvent_data_test = solvent_data_test.reshape(-1, solvent_data_test.shape[-1])
        init_solvent_test = self.fc1(solvent_data_test)
        # 溶剂进行2层GraphSAGE+resnet
        solvent_data_test = F.relu(self.solvent_conv1(init_solvent_test, solvent_test["solvent_adj"]))
        solvent_data_test = self.solvent_conv2(solvent_data_test, solvent_test["solvent_adj"]) + init_solvent_test

        solute_batch = self.get_graph_pool_batch(len_solute).to(self.DEVICE)
        solvent_test_batch = self.get_graph_pool_batch(len_solvent_test).to(self.DEVICE)

        solute_test = self.set2set(solute_data_test, solute_batch)
        solvent_test = self.set2set(solvent_data_test, solvent_test_batch)

        data0 = torch.cat((solute_test, after_solute_smile_vectors), 1)
        data1 = torch.cat((solvent_test, after_test_solvent_smile_vectors), 1)
        data1 = torch.cat((data0, data1), 1)
        data1 = self.dropout(F.relu(self.fc2(data1)))
        data1 = self.dropout(F.relu(self.fc3(data1)))
        data1 = self.dropout(F.relu(self.fc4(data1)))
        data1 = self.fc5(data1)

        return data1
