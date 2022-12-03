from torch import nn
import torch.nn.functional as F

class MyNet9(nn.Module):
    def __init__(self):
        super(MyNet9, self).__init__()
        # (B, 170, 80, 280)
        self.conv1 = nn.Conv2d(170, 128, 3, stride=1, padding=1)
        self.batchnormalization1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.batchnormalization2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64*20*70, 180)
        self.fc2 = nn.Linear(180, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnormalization1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batchnormalization2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features