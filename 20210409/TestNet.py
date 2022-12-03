import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print("x.shape1=",x.shape)
        x = self.conv1(x)
        print("x.shape2=", x.shape)
        x = F.relu(x)
        print("x.shape3=", x.shape)
        x = F.max_pool2d(x,(2,2))
       # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print("x.shape4=", x.shape)
        # If the size is a square you can only specify a single number

        x = self.conv2(x)
        print("x.shape5=", x.shape)
        x = F.relu(x)
        print("x.shape6=", x.shape)
        x = F.max_pool2d(x, (2, 2))
       # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print("x.shape7=", x.shape) #(1,16,6,6)
        x = x.view(-1, self.num_flat_features(x))
        print("x.shape8=", x.shape)
        x = F.relu(self.fc1(x))
        print("x.shape9=", x.shape)
        x = F.relu(self.fc2(x))
        print("x.shape10=", x.shape)
        x = self.fc3(x)
        print("x.shape11=", x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out.shape)