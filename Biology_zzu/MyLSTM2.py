#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import GraphConvolution

class MyLSTM2(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size, 3)

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

    def forward(self, x, h1, c1):
        out,(h1,c1) = self.lstm(x, (h1, c1))
        # print("out1.shape=",out.shape)
        a,b,c = out.shape
        out = self.fc(out.reshape(-1,c))
        out = out.reshape(a,b,-1)
        # print("out2.shape=", out.shape)
        # print("out=",out)
        # print("out[:,-1,:]=", out[:,-1,:].shape)
        # print("out.shape=",out.shape)(1,100,3)
        return out,(h1,c1)