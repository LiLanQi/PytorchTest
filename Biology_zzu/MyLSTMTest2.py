#encoding=utf-8
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os
from MyLSTM2 import MyLSTM2
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--result_src', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)

DEVICE = torch.device('cuda:1')
args = parser.parse_args()
writer = SummaryWriter('MyLSTMTest2')

dict_solute = {}
dict_solvent = {}

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.numpy()
    rowsum = np.array(mx.sum(1))             # 对每一个特征进行归一化
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.from_numpy(mx)
    return mx

def get_solute_position(rootdir):
    label_index = [1]
    path_list = readFileData(rootdir,label_index)
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        for i in range(5, 173 + 5):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
            dict_solute[temp_atom] = i-5
        break
    return dict_solute

def get_solvent_position(rootdir):
    label_index = [1]
    path_list = readFileData(rootdir,label_index)
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        for i in range(len(file_list) - 8, len(file_list) - 2):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
            dict_solvent[temp_atom] = i-(len(file_list) - 8)
        break
    return dict_solvent


# 返回一个结果张量
def readFileResult(root_dir):
    file = open(root_dir, "r")
    file_list = file.readlines()
    result_list = []
    for i in range(0, 100000):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result
total_labels = readFileResult(args.result_src)

def readFileData(root_dir, lable_index):
    path_list = []
    for i in range(len(lable_index)):
        j = lable_index[i]
        temp_path = "ACE" + str(j) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list

def get_solute_adj():
    solute_adj = torch.zeros(173*12,173*12)
    bonds = 'S1-O3', 'S1-O4', 'S1-F1', 'S1-O1', 'O1-C3', 'C3-C4', 'C4-H3', 'C4-C5', 'C5-H4', 'C5-C6', 'C6-C1', 'C1-H1', 'C1-C2', 'C2-H2', 'C2-C3', 'C6-C19', 'C19-C14', 'C14-C13', 'C13-H9', 'C13-C18', 'C18-H12', 'C18-C17', 'C17-C16', 'C16-H11', 'C16-C15', 'C15-H10', 'C15-C14', 'C17-O2', 'O2-S2', 'S2-O5', 'S2-O6', 'S2-F2', 'C19-H13', 'C19-C10', 'C10-C11', 'C11-H7', 'C11-C12', 'C12-H8', 'C12-C7', 'C7-C8', 'C8-H5', 'C8-C9', 'C9-H6', 'C9-C10', 'C7-O25', 'O25-S8', 'S8-O26', 'S8-O27', 'S8-O22', 'O22-C64', 'C64-C69', 'C69-H47', 'C69-C68', 'C68-H46', 'C68-C67', 'C67-C66', 'C66-H45', 'C66-C65', 'C65-H44', 'C65-C64', 'C67-C76', 'C76-H52', 'C76-C71', 'C71-C70', 'C70-H48', 'C70-C75', 'C75-H51', 'C75-C74', 'C74-C73', 'C73-H50', 'C73-C72', 'C72-H49', 'C72-C71', 'C74-O24', 'O24-S9', 'S9-O29', 'S9-O30', 'S9-O28', 'O28-C26', 'C26-C27', 'C27-H18', 'C27-C28', 'C28-H19', 'C28-C29', 'C29-C30', 'C30-H20', 'C30-C31', 'C31-H21', 'C31-C26', 'C29-C38', 'C38-H26', 'C38-C25', 'C25-C20', 'C20-H14', 'C20-C21', 'C21-H15', 'C21-C22', 'C22-C23', 'C23-H16', 'C23-C24', 'C24-H17', 'C24-C25', 'C22-O7', 'O7-S3', 'S3-O9', 'S3-O10', 'S3-F3', 'C38-C33', 'C33-C32', 'C32-H22', 'C32-C37', 'C37-H25', 'C37-C36', 'C36-C35', 'C35-H24', 'C35-C34', 'C34-H23', 'C34-C33', 'C36-O8', 'O8-S4', 'S4-O11', 'S4-O12', 'S4-F4', 'C76-C63', 'C63-C62', 'C62-H43', 'C62-C61', 'C61-H42', 'C61-C60', 'C60-C59', 'C59-H41', 'C59-C58', 'C58-H40', 'C58-C63', 'C60-O23', 'O23-S7', 'S7-O20', 'S7-O21', 'S7-O17', 'O17-C55', 'C55-C56', 'C56-H38', 'C56-C51', 'C51-H35', 'C51-C52', 'C52-C53', 'C53-H36', 'C53-C54', 'C54-C55', 'C54-H37', 'C52-C57', 'C57-H39', 'C57-C48', 'C48-C47', 'C47-H32', 'C47-C46', 'C46-H31', 'C46-C45', 'C45-C50', 'C50-H34', 'C50-C49', 'C49-H33', 'C49-C48', 'C45-O13', 'O13-S5', 'S5-O14', 'S5-O15', 'S5-F5', 'C57-C44', 'C44-C43', 'C43-H30', 'C43-C42', 'C42-H29', 'C42-C41', 'C41-C40', 'C40-H28', 'C40-C39', 'C39-H27', 'C39-C44', 'C41-O16', 'O16-S6', 'S6-O18', 'S6-O19', 'S6-F6'
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(12):
            left_atom_value = dict_solute[left_atom] + 173 * i
            right_atom_value = dict_solute[right_atom] + 173 * i
            solute_adj[left_atom_value][right_atom_value] = 1
            solute_adj[right_atom_value][left_atom_value] = 1
    solute_adj = normalize(solute_adj + torch.eye(2076, 2076))
    return solute_adj

def get_solvent_adj():
    solvent_adj = torch.zeros(6*1490,6*1490)
    bonds = 'C1-H1', 'C1-H2', 'C1-H3', 'C1-C2', 'C2-N1'
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(1490):
            left_atom_value = dict_solvent[left_atom] + 6 * i
            right_atom_value = dict_solvent[right_atom] + 6 * i
            solvent_adj[left_atom_value][right_atom_value] = 1
            solvent_adj[right_atom_value][left_atom_value] = 1
    solvent_adj =  normalize(solvent_adj + torch.eye(6*1490, 6*1490))
    return solvent_adj



class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase, lable_index):
        self.root = rootdir
        path_list = readFileData(rootdir, lable_index)  # 得到对应ACEx.pdb文件的list
        # 对所有的ACE.pdb文件里的数据进行预处理
        current_data = 0
        temp_result_list = []
        for path in path_list:
            current_data = current_data + 1
            if(current_data%100 == 9):
                print("path=",path,"current_data=",current_data,",all_data=",len(path_list),"")
            file = open(path, "r")
            file_list = file.readlines()
            for i in range(5, 6):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                temp_result_list.append(x)
                temp_result_list.append(y)
                temp_result_list.append(z)
        temp = torch.tensor(temp_result_list).reshape(-1, 3)
        print("temp.shape=",temp.shape)
        self.data = temp

    def __getitem__(self, index):
        input_data  = self.data[index:index + 100,:]
        output_data = self.data[index+1:index + 100+1,:]
        return input_data, output_data

    def __len__(self):
        return len(self.data) - 100


def load_train(root_path, batch_size, result_src, phase, label_index):
    data = trainset(root_path, result_src, phase, label_index)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=12)
    return loader


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch / num_epochs) ** power


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_text_data(rootdir, result_src, lable_index):
    path_list = readFileData(rootdir, lable_index)  # 得到对应ACEx.pdb文件的list
    # 对所有的ACE.pdb文件里的数据进行预处理
    current_data = 0
    temp_result_list = []
    for path in path_list:
        current_data = current_data + 1
        if(current_data%100 == 9):
            print("path=",path,"current_data=",current_data,",all_data=",len(path_list),"")
        file = open(path, "r")
        file_list = file.readlines()
        for i in range(5, 6):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            x = float(temp_data[0] + "." + temp_data[1][0:3])
            y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
            z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
            temp_result_list.append(x)
            temp_result_list.append(y)
            temp_result_list.append(z)
    text_data = torch.tensor(temp_result_list).reshape(-1, 3)
    return text_data

if __name__ == '__main__':
    data_list = []
    for i in range(1000):
        data_list.append(i)
    train_lable_index = data_list

    train_loader = load_train(args.data_src, args.batchsize, args.result_src, "train",train_lable_index)

    rnn = MyLSTM2(input_size=3, hidden_size=250, num_layers=1)
    rnn = rnn.to(DEVICE)
    # rnn.apply(weight_init)
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = torch.nn.MSELoss()
    h1 = Variable(torch.zeros(1, args.batchsize, 250))
    c1 = Variable(torch.zeros(1, args.batchsize, 250))
    h1 = h1.to(DEVICE)
    c1 = c1.to(DEVICE)
    for epoch in range(1):
        total_loss = 0
        for i_batch, batch_data in tqdm(enumerate(train_loader)):

            input_data, label_data = batch_data
            # print("inputx_data=",inputx_data)
            # print("outputx_data=", outputx_data)

            input_data = input_data.to(DEVICE)
            label_data = label_data.to(DEVICE) #(1,100,3)
            optimizer.zero_grad()
            outputs, (h1,c1) = rnn(input_data,(h1,c1))
            # print("outputs=", outputs)
            # print("(h1,c1) =", (h1, c1))
            loss = criterion(outputs, label_data)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item() * outputs.size(0)
        writer.add_scalar('train loss', total_loss / len(train_loader), epoch)
    print("len(train_loader)=",len(train_loader)) #225
    PATH1 = './MyLSTMest2_net.pth'
    torch.save(rnn.state_dict(), PATH1)

    #test
    rnn.eval()
    test_list = []
    for i in range(900,1300):
        test_list.append(i)
    test_lable_index = test_list
    test_all_data = get_text_data(args.data_src,args.result_src, test_lable_index)
    # print("test_all_data=",test_all_data)
    test_data = test_all_data[0:100,:].unsqueeze(0) #已经使用过
    # print("test_data1.shape=", test_data.shape)
    label_data = test_all_data[101:,:] #真正的label
    # print("label_data2.shape=", label_data.shape)
    total_loss = 0
    prediction_data = torch.zeros(1,100,3)
    for i_batch in range(1000,1200):
        test_data = test_data.to(DEVICE)
        if(i_batch != 1000):
            test_data = torch.cat((test_data[:,1:,:],prediction_data[:,-1,:].unsqueeze(0)), 1)
        test_correct_label = label_data[i_batch-1000:i_batch-1000+100,:].unsqueeze(0).to(DEVICE)
        with torch.set_grad_enabled(False):
            outputs, (h1,c1) = rnn(test_data,(h1,c1))
            loss = criterion(outputs, test_correct_label)
        prediction_data = outputs
        prediction_data = prediction_data.to(DEVICE)
        deviation = (abs(outputs - test_correct_label) / abs(test_correct_label)).squeeze(0)
        print("outputs=", outputs)
        print("labels=", test_correct_label)
        print("outputs-output_data=", abs(outputs - test_correct_label))
        print("deviation=", deviation)
        print("loss=", loss.item())
        writer.add_scalar('predic loss', loss.item(), i_batch-1000)



