from torch import nn
import torch.nn.functional as F

class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        #(B,80,160,70)
        self.conv1 = nn.Conv2d(80, 128, 3, stride=1, padding=1)
        self.batchnormalization1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.batchnormalization2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batchnormalization3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.batchnormalization4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.batchnormalization5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.batchnormalization6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.batchnormalization7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batchnormalization8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batchnormalization9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.batchnormalization10 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)


        self.fc1 = nn.Linear(512*20*8, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnormalization1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnormalization2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnormalization3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchnormalization4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.batchnormalization5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batchnormalization6(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.batchnormalization7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.batchnormalization8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.batchnormalization9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.batchnormalization10(x)
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