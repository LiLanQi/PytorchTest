#encoding=utf-8
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os
from MyLastGCNTestWAT import MyNewGCN
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#可以适合任何溶剂+溶质
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--result_src_NMF', type=str, default='../data/newNMF/NMF-LJ.xvg')
parser.add_argument('--data_src_NMF', type=str, default='../data/newNMF/NMF_pdb')
parser.add_argument('--result_src_ACE', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src_ACE', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--result_src_wat', type=str, default='../data/newWater/water-LJ.xvg')
parser.add_argument('--data_src_wat', type=str, default='../data/newWater/WAT_pdb')
parser.add_argument('--result_src_meth', type=str, default='../data/newMeth/meth-LJ.xvg')
parser.add_argument('--data_src_meth', type=str, default='../data/newMeth/meth_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0)

DEVICE = torch.device('cuda:0')
args = parser.parse_args()
writer = SummaryWriter('MyLastGCNTest')

dict_solute = {}
dict_solvent_ACE = {}
dict_solvent_NMF = {}
dict_solvent_wat = {}
dict_solvent_meth = {}

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
    path_list = readFileData(rootdir,label_index, "ACE")
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

def get_solvent_position(rootdir, solvent):
    dict_solvent = {}
    if (solvent == "ACE"):
        botom_index = 8
    elif (solvent == "NMF"):
        botom_index = 11
    elif (solvent == "wat"):
        botom_index = 5
    elif (solvent == "meth"):
        botom_index = 17
    label_index = [1]
    path_list = readFileData(rootdir,label_index, solvent)
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        left = len(file_list) - botom_index
        right = len(file_list) - 2
        for i in range(left, right):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
            dict_solvent[temp_atom] = i-left
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
total_labels_ACE = readFileResult(args.result_src_ACE)
total_labels_NMF = readFileResult(args.result_src_NMF)
total_labels_wat = readFileResult(args.result_src_wat)
total_labels_meth = readFileResult(args.result_src_meth)

def readFileData(root_dir, lable_index, solvent):
    path_list = []
    for i in range(len(lable_index)):
        j = lable_index[i]
        temp_path = solvent + str(j) + ".pdb"
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
    solute_adj = solute_adj + torch.eye(2076, 2076)
    return solute_adj

def get_solvent_adj(solvent, dim):
    if (solvent == "ACE"):
        atom_nums = 6
        atom_g = 1490
        bonds = 'C1-H1', 'C1-H2', 'C1-H3', 'C1-C2', 'C2-N1'
        dict_solvent = dict_solvent_ACE
    elif (solvent == "NMF"):
        atom_nums = 9
        atom_g = 1350
        bonds = 'C1-O1', 'C1-H1', 'C1-N1', 'N1-H2', 'C2-N1', 'C2-H3', 'C2-H4', 'C2-H5'
        dict_solvent = dict_solvent_NMF
    elif (solvent == "wat"):
        atom_nums = 3
        atom_g = 4928
        bonds = 'O-H1', 'O-H2'
        dict_solvent = dict_solvent_wat
    elif (solvent == "meth"):
        atom_nums = 15
        atom_g = 1089
        bonds = 'C7-H6', 'C7-H7','C7-H8','C7-C2','C2-C1','C1-H1','C1-C6','C6-H5','C6-C5','C5-H4','C5-C4','C4-H3','C4-C3','C3-H2','C3-C2'
        dict_solvent = dict_solvent_meth
    solvent_adj = torch.zeros(dim,dim)
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(atom_g):
            left_atom_value = dict_solvent[left_atom] + atom_nums * i
            right_atom_value = dict_solvent[right_atom] + atom_nums * i
            solvent_adj[left_atom_value][right_atom_value] = 1
            solvent_adj[right_atom_value][left_atom_value] = 1
    A_loop = torch.eye(atom_nums*atom_g, atom_nums*atom_g)
    solvent_adj =  solvent_adj + A_loop
    return solvent_adj




class trainset(Dataset):
    def __init__(self, rootdir_ACE, result_src_ACE, phase, rootdir_NMF, result_src_NMF, lable_index, rootdir_wat, rootdir_meth, solute_addtional_feature, wat_atom_to_feature):
        self.root = rootdir_ACE
        path_list_wat = readFileData(rootdir_wat, lable_index, "wat")  # 得到对应watx.pdb文件的list
        # atoms_dict = {'H': 1, 'C': 6, 'O': 8, 'N': 7, 'S': 16,
        #               'F': 9}  # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274

        solute_list_wat = []
        solvent_list_wat = []
        lables_list_wat = []
        current_data = 0
        for path in path_list_wat:
            current_data = current_data + 1
            if (current_data % 100 == 9):
                print("path=", path, "current_data=", current_data, ",all_data=", len(path_list_wat), "")
            file = open(path, "r")
            file_list = file.readlines()
            temp_result_list = []
            position_list = []
            for i in range(5, len(file_list) - 2):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
                # single_temp_atom = file_list[i][13]
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # print("solute_addtional_feature[temp_atom]=",solute_addtional_feature[temp_atom].shape)
                if (i < 2081):
                    temp_result_list.append(solute_addtional_feature[temp_atom])
                else:
                    temp_result_list.append(wat_atom_to_feature[temp_atom])
                position_list.append(x)
                position_list.append(y)
                position_list.append(z)

            temp = torch.stack(temp_result_list)
            position_tensor = torch.tensor(position_list).reshape(-1, 3)
            temp = torch.cat((temp, position_tensor), 1)
            solute_data = temp[:173 * 12, :]
            solvent_data = temp[173 * 12:, :]
            solute_list_wat.append(solute_data)  # 存入每个wat文件的溶质图
            solvent_list_wat.append(solvent_data)  # 存入每个wat文件的溶剂图
        for i in range(len(lable_index)):
            lables_list_wat.append(total_labels_wat[lable_index[i]])  # 存入每个wat文件的标签
        self.solute_list_wat = solute_list_wat
        self.solvent_list_wat = solvent_list_wat
        self.lables_list_wat = lables_list_wat  # 存入所有mdx 文件对应的output值

    def __getitem__(self, index):

        solute_list_wat = self.solute_list_wat[index]
        solvent_list_wat = self.solvent_list_wat[index]
        lable_wat = self.lables_list_wat[index]

        return solute_list_wat, solvent_list_wat, lable_wat

    def __len__(self):
        return len(self.solute_list_wat)


def load_train(root_path_ACE, batch_size, result_src_ACE, phase,root_path_NMF,result_src_NMF, label_index, root_path_wat, root_path_meth, solute_addtional_feature, wat_atom_to_feature):
    data = trainset(root_path_ACE, result_src_ACE, phase, root_path_NMF, result_src_NMF, label_index, root_path_wat, root_path_meth, solute_addtional_feature, wat_atom_to_feature)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader

if __name__ == '__main__':
    test_lable_index = []
    for i in range(100):
        test_lable_index.append(i)
    dict_solute = get_solute_position(args.data_src_ACE)
    dict_solvent_NMF = get_solvent_position(args.data_src_NMF, "NMF")
    dict_solvent_ACE = get_solvent_position(args.data_src_ACE, "ACE")
    dict_solvent_wat = get_solvent_position(args.data_src_wat, "wat")
    dict_solvent_meth = get_solvent_position(args.data_src_meth, "meth")
    solute_adj = get_solute_adj()
    solvent_adj_NMF = get_solvent_adj("NMF",12150)
    solvent_adj_ACE = get_solvent_adj("ACE", 8940)
    solvent_adj_wat = get_solvent_adj("wat", 14784)
    solvent_adj_meth = get_solvent_adj("meth", 16335)
    meth_atom_to_feature = np.load("./meth_atom_to_feature.npy", allow_pickle=True).item()
    wat_atom_to_feature = np.load("./wat_atom_to_feature.npy", allow_pickle=True).item()
    NMF_atom_to_feature = np.load("./NMF_atom_to_feature.npy", allow_pickle=True).item()
    ACE_atom_to_feature = np.load("./ACE_atom_to_feature.npy", allow_pickle=True).item()
    solute_addtional_feature = np.load("./solute_atom_to_feature.npy", allow_pickle=True).item()
    net = MyNewGCN(nfeat=9,nhid=64,nclass=128,dropout=0)
    net = net.to(DEVICE)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = torch.nn.MSELoss()
    loss_list = []

    PATH1 = './check_point/1MyLastGCNTest8_net.pth'
    net.load_state_dict(torch.load(PATH1))


    #test
    net.eval()
    test_loader = load_train(args.data_src_ACE, args.batchsize, args.result_src_ACE, "test", args.data_src_NMF,
                              args.result_src_NMF, test_lable_index, args.data_src_wat, args.data_src_meth,
                             solute_addtional_feature, wat_atom_to_feature)
    SUMwat1 = 0
    SUMwat2 = 0
    SUMwat3 = 0
    SUMwat4 = 0
    SUMwat5 = 0
    SUMwat6 = 0
    SUMwat7 = 0
    SUMwat8 = 0
    SUMwat9 = 0
    SUMwat10 = 0
    for i_batch, batch_data in tqdm(enumerate(test_loader)):
        solute_data_wat, solvent_data_wat, labels_wat = batch_data
        labels_wat = labels_wat.to(DEVICE)
        labels = labels_wat
        solute_data_wat = solute_data_wat.to(DEVICE)
        solvent_data_wat = solvent_data_wat.to(DEVICE)
        solute_adj = solute_adj.to(DEVICE)
        solvent_adj_wat = solvent_adj_wat.to(DEVICE)
        with torch.set_grad_enabled(False):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
            outputs = net(solute_adj,solute_data_wat,
                          solvent_data_wat, solvent_adj_wat).squeeze(-1)
        deviation = abs((outputs - labels) / labels)
        print("outputs=", outputs)
        print("labels=", labels)
        print("deviation=", deviation)
        for i in range(deviation.shape[0]):
            if (i <= 3):
                if (deviation[i] < 0.01):
                    SUMwat1 = SUMwat1 + 1
                if (deviation[i] < 0.02):
                    SUMwat2 = SUMwat2 + 1
                if (deviation[i] < 0.03):
                    SUMwat3 = SUMwat3 + 1
                if (deviation[i] < 0.04):
                    SUMwat4 = SUMwat4 + 1
                if (deviation[i] < 0.05):
                    SUMwat5 = SUMwat5 + 1
                if (deviation[i] < 0.06):
                    SUMwat6 = SUMwat6 + 1
                if (deviation[i] < 0.07):
                    SUMwat7 = SUMwat7 + 1
                if (deviation[i] < 0.08):
                    SUMwat8 = SUMwat8 + 1
                if (deviation[i] < 0.09):
                    SUMwat9 = SUMwat9 + 1
                if (deviation[i] < 0.1):
                    SUMwat10 = SUMwat10 + 1

    print("SUMwat1 = ", SUMwat1, "SUMwat2 = ", SUMwat2, "SUMwat3 = ", SUMwat3, "SUMwat4 = ", SUMwat4, "SUMwat5 = ", SUMwat5)
    print("SUMwat6 = ", SUMwat6, "SUMwat7 = ", SUMwat7, "SUMwat8 = ", SUMwat8, "SUMwat9 = ", SUMwat9, "SUMwat10 = ", SUMwat10)
    print("tatol", SUMwat5, " wat in 5%")




