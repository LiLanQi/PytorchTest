# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from dgl._deprecate.graph import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

from utils import get_graph_from_smile
from testmodel import DataEmbedding


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


    # (nfeat=3,nhid=8,nclass=16,dropout=0.5,solute_solvent_size=溶质第0维+溶剂第0维)
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyNewGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.fc1 = nn.Linear(32 + 69, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.embedding = DataEmbedding()

        self.dropout = dropout
        self.graph_pool_solute_ACE, self.graph_pool_solvent_ACE = self.get_graph_pool_ACE()
        self.graph_pool_solute_NMF, self.graph_pool_solvent_NMF = self.get_graph_pool_NMF()
        self.graph_pool_solute_wat, self.graph_pool_solvent_wat = self.get_graph_pool_wat()
        self.DEVICE = torch.device('cuda:0')

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute_ACE, solvent_ACE, solute_adj, solvent_adj_ACE, solute_NMF, solvent_NMF, solvent_adj_NMF, solute_wat, solvent_wat, solvent_adj_wat, solute_enviroment_data, ACE_solvent_enviroment_data, NMF_solvent_enviroment_data, wat_solvent_enviroment_data):
        graph_pool_solute_ACE = self.graph_pool_solute_ACE.to(self.DEVICE)
        graph_pool_solvent_ACE = self.graph_pool_solvent_ACE.to(self.DEVICE)
        graph_pool_solute_NMF = self.graph_pool_solute_NMF.to(self.DEVICE)
        graph_pool_solvent_NMF = self.graph_pool_solvent_NMF.to(self.DEVICE)
        graph_pool_solute_wat = self.graph_pool_solute_wat.to(self.DEVICE)
        graph_pool_solvent_wat = self.graph_pool_solvent_wat.to(self.DEVICE)

        solute_ACE = F.relu(self.gc1(solute_ACE, solute_adj))
        solute_ACE = self.gc2(solute_ACE, solute_adj)
        solute_NMF = F.relu(self.gc1(solute_NMF, solute_adj))
        solute_NMF = self.gc2(solute_NMF, solute_adj)
        solute_wat = F.relu(self.gc1(solute_wat, solute_adj))
        solute_wat = self.gc2(solute_wat, solute_adj)

        # print("solute.shape=",solute.shape)#(4, 2076, 16))
        solvent_ACE = F.relu(self.gc1(solvent_ACE, solvent_adj_ACE))
        solvent_ACE = self.gc2(solvent_ACE, solvent_adj_ACE)
        solvent_NMF = F.relu(self.gc1(solvent_NMF, solvent_adj_NMF))
        solvent_NMF = self.gc2(solvent_NMF, solvent_adj_NMF)
        solvent_wat = F.relu(self.gc1(solvent_wat, solvent_adj_wat))
        solvent_wat = self.gc2(solvent_wat, solvent_adj_wat)
        # print("solvent.shape=", solvent.shape)#(4, 8940, 16))

        solute_ACE = solute_ACE.reshape(-1, 16)
        solvent_ACE = solvent_ACE.reshape(-1, 16)
        solute_NMF = solute_NMF.reshape(-1, 16)
        solvent_NMF = solvent_NMF.reshape(-1, 16)
        solute_wat = solute_wat.reshape(-1, 16)
        solvent_wat = solvent_wat.reshape(-1, 16)

        # print("solute.shape=",solute.shape, "graph_pool_solute.shape=", graph_pool_solute.shape)
        solute_ACE = torch.mm(graph_pool_solute_ACE, solute_ACE)
        solvent_ACE = torch.mm(graph_pool_solvent_ACE, solvent_ACE)
        solute_NMF = torch.mm(graph_pool_solute_NMF, solute_NMF)
        solvent_NMF = torch.mm(graph_pool_solvent_NMF, solvent_NMF)
        solute_wat = torch.mm(graph_pool_solute_wat, solute_wat)
        solvent_wat = torch.mm(graph_pool_solvent_wat, solvent_wat)

        solute_ACE_information = torch.cat((solute_enviroment_data, ACE_solvent_enviroment_data), 1)
        solute_ACE_information = torch.cat((solute_ACE_information, solute_ACE_information), 0)
        solute_ACE_information = torch.cat((solute_ACE_information, solute_ACE_information), 0)

        solute_NMF_information = torch.cat((solute_enviroment_data, NMF_solvent_enviroment_data), 1)
        solute_NMF_information = torch.cat((solute_NMF_information, solute_NMF_information), 0)
        solute_NMF_information = torch.cat((solute_NMF_information, solute_NMF_information), 0)

        solute_wat_information = torch.cat((solute_enviroment_data, wat_solvent_enviroment_data), 1)
        solute_wat_information = torch.cat((solute_wat_information, solute_wat_information), 0)
        solute_wat_information = torch.cat((solute_wat_information, solute_wat_information), 0)

        # 溶剂偶极矩、分子体表面积、分子体积,(batchsize=4,5)
        ACE_another_feature = torch.tensor([[-1.3711368, -0.813852, -0.0000731, 65.13210, 84.32505],
                                                    [-1.3711368, -0.813852, -0.0000731, 65.13210, 84.32505],
                                                    [-1.3711368, -0.813852, -0.0000731, 65.13210, 84.32505],
                                                    [-1.3711368, -0.813852, -0.0000731, 65.13210, 84.32505]])
        NMF_another_feature = torch.tensor([[-1.6522485, -0.2748812, -0.0000781, 83.89315, 101.94061],
                                                    [-1.6522485, -0.2748812, -0.0000781, 83.89315, 101.94061],
                                                    [-1.6522485, -0.2748812, -0.0000781, 83.89315, 101.94061],
                                                    [-1.6522485, -0.2748812, -0.0000781, 83.89315, 101.94061]])
        wat_another_feature = torch.tensor([[0.4208888, 0.5955439, 0., 28.78463, 45.70048],
                                                    [0.4208888, 0.5955439, 0., 28.78463, 45.70048],
                                                    [0.4208888, 0.5955439, 0., 28.78463, 45.70048],
                                                    [0.4208888, 0.5955439, 0., 28.78463, 45.70048]])
        #合并其他的feature
        ACE_another_feature = torch.cat((solute_ACE_information, ACE_another_feature), 1)
        NMF_another_feature = torch.cat((solute_NMF_information, NMF_another_feature), 1)
        wat_another_feature = torch.cat((solute_wat_information, wat_another_feature), 1)
        ACE_another_feature = ACE_another_feature.to(self.DEVICE)
        NMF_another_feature = NMF_another_feature.to(self.DEVICE)
        wat_another_feature = wat_another_feature.to(self.DEVICE)

        data1 = torch.cat((solute_ACE, solvent_ACE), 1)
        data1 = torch.cat((data1, ACE_another_feature), 1)
        data1 = F.relu(self.fc1(data1))
        data1 = F.relu(self.fc2(data1))
        data1 = F.relu(self.fc3(data1))
        data1 = self.fc4(data1)

        data2 = torch.cat((solute_NMF, solvent_NMF), 1)
        data2 = torch.cat((data2, NMF_another_feature), 1)
        data2 = F.relu(self.fc1(data2))
        data2 = F.relu(self.fc2(data2))
        data2 = F.relu(self.fc3(data2))
        data2 = self.fc4(data2)

        data3 = torch.cat((solute_wat, solvent_wat), 1)
        data3 = torch.cat((data3, wat_another_feature), 1)
        data3 = F.relu(self.fc1(data3))
        data3 = F.relu(self.fc2(data3))
        data3 = F.relu(self.fc3(data3))
        data3 = self.fc4(data3)

        data = torch.cat((data1, data2), 0)
        data = torch.cat((data, data3), 0)
        return data