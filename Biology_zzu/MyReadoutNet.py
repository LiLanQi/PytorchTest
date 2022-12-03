from torch import nn
import torch.nn.functional as F

class MyReadoutNet(nn.Module):
    def __init__(self):
        super(MyReadoutNet, self).__init__()
        #(4,11016,16)
        self.fc1 = nn.Linear(11016*16, 360)
        self.fc2 = nn.Linear(360, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        print("x.shape1=",x.shape)
        x = x.view(-1, self.num_flat_features(x))
        print("x.shape2=", x.shape)
        x = self.fc1(x)
        print("x.shape3=", x.shape)
        x = F.relu(x)
        print("x.shape4=", x.shape)
        x = self.dropout(x)
        print("x.shape5=", x.shape)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features