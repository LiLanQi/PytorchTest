#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set, NNConv, SAGEConv


from layers import GraphConvolution


class GatherModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True
                           )

    def forward(self, g, n_feat, e_feat):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : node features
        """

        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g, out, e_feat))
            else:
                m = torch.relu(self.conv.bias +  self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        # print("init.shape=", init.shape, "out.shape=", out.shape)
        return out + init



#新的网络，可以容纳任何溶质
class MyCIGCN(nn.Module):
    #(node_input_dim=4,edge_input_dim=1,node_hidden_dim=42,edge_hidden_dim=42,num_step_message_passing=6)
    def __init__(self, node_input_dim, edge_input_dim, node_hidden_dim, edge_hidden_dim, num_step_message_passing):
        super(MyCIGCN, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_hidden_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_hidden_dim,
                                          self.num_step_message_passing,
                                          )


        self.set2set_solute = Set2Set(2 * node_hidden_dim, 2, 1)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, 2, 1)

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        #
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        # self.fc2 = nn.Linear(8, 1)
        #
        # self.dropout = dropout

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, solute, solvent, solute_len_matrix, solvent_len_matrix):
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        # print("solute_features=",solute_features)
        # print("solute_features.shape1=",solute_features.shape) #(8304,4)
        solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        # print("solvent_features=", solvent_features)
        # print("solvent_features.shape1=", solvent_features.shape)#(35760,4)
        # print("solute=",solute)
        # print("solvent", solvent)
        # print("solute_len_matrix.shape=",solute_len_matrix.shape,"solvent_len_matrix.shape=",solvent_len_matrix.shape)
        len_map = torch.mm(solute_len_matrix.t(), solvent_len_matrix) #[4, 8304] [4, 35760] = (8304,35760)
        interaction_map = torch.mm(solute_features, solvent_features.t())#(8304, 4)(4, 35760) = (8304,35760)
        interaction_map = torch.tanh(interaction_map)
        interaction_map = torch.mul(len_map.float(), interaction_map) #(8304,35760)
        solvent_prime = torch.mm(interaction_map.t(), solute_features)  # (35760,8304)*(8304,4) = (35760,4)
        solute_prime = torch.mm(interaction_map, solvent_features)  #(8304,35760)*(35760,4) = (8304, 4)

        solute_features = torch.cat((solute_features, solute_prime), dim=1)  #(8304, 8)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)  #(35760,8)

        solute_features = self.set2set_solute(solute, solute_features)#(4, 16)
        solvent_features = self.set2set_solvent(solvent, solvent_features)#(4, 16)
        # print("solvent_features=", solvent_features.shape)
        # print("solute_features=", solute_features.shape, "solvent_features=", solvent_features.shape)#(4,8)
        data = torch.cat((solute_features, solvent_features), 1) #(4,32)
        # print("data.shape=",data.shape)
        data = F.relu(self.fc1(data))
        data = self.fc2(data)
        # print("data=",data)
        # data = F.elu(data)
        return data