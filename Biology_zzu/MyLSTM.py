#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import GraphConvolution

class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
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

    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        # print("out1.shape=",out.shape)
        a,b,c = out.shape
        out = self.fc(out.reshape(-1,c))
        out = out.reshape(a,b,-1)
        # print("out2.shape=", out.shape)
        # print("out=",out)
        # print("out[:,-1,:]=", out[:,-1,:].shape)
        return out[:,-1,:].unsqueeze(0)