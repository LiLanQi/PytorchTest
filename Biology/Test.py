import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 24)
        self.conv1 = GCNConv(24, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print("x1.shape=", x.shape)
        x = self.fc1(x.long())
        print("x2.shape=", x.shape)
        x = self.conv1(x, edge_index)
        print("x3.shape=", x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
# data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    for batch in loader:
        print(batch)
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
