import random

import numpy as np
from torch import nn
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os

from warmup_scheduler import GradualWarmupScheduler

from MyFinalSuperGAT4 import MyNewGNN, MyValModel
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--result_src_NMF', type=str, default='../data/newNMF/NMF-LJ.xvg')
parser.add_argument('--data_src_NMF', type=str, default='../data/newNMF/NMF_pdb')
parser.add_argument('--result_src_ACE', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src_ACE', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--result_src_wat', type=str, default='../data/newWater/water-LJ.xvg')
parser.add_argument('--data_src_wat', type=str, default='../data/newWater/WAT_pdb')
parser.add_argument('--result_src_meth', type=str, default='../data/newMeth/meth-LJ.xvg')
parser.add_argument('--data_src_meth', type=str, default='../data/newMeth/meth_pdb')
parser.add_argument('--result_src_DMF', type=str, default='../data/newDMF/DMF-LJ.xvg')
parser.add_argument('--data_src_DMF', type=str, default='../data/newDMF/DMF_pdb')
parser.add_argument('--base_lr', type=float, default=1E-5)
# parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nfeat', type=int, default=9)
parser.add_argument('--nhid', type=int, default=64)
parser.add_argument('--nclass', type=int, default=64)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--total_dataset', type=int, default=5)
parser.add_argument('--test_index', type=int, default=4)
parser.add_argument('--total_data', type=int, default=100000)
parser.add_argument('--warmup_step', type=int, default=10)
parser.add_argument('--patience', type=int, default=50)

# parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

dict_solute = {}
dict_solvent_NMF = {}
dict_solvent_wat = {}
dict_solvent_meth = {}
dict_solvent_DMF = {}

DEVICE = torch.device('cuda:0')
torch.cuda.set_device(0)
args = parser.parse_args(args=[])

writer = SummaryWriter('MySuperGATTest4')


def adjust_learning_rate(optimizer, iter, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {iter: args.learning_rate * (0.5 ** ((iter - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if iter in lr_adjust.keys():
        lr = lr_adjust[iter]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.numpy()
    rowsum = np.array(mx.sum(1))  # normalize
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.from_numpy(mx)
    return mx


def get_solute_position(rootdir):
    label_index = [1]
    path_list = readFileData(rootdir, label_index, "ACE")
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        for i in range(5, 173 + 5):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # current atom(include number)
            dict_solute[temp_atom] = i - 5
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
    elif (solvent == "DMF"):
        botom_index = 14
    label_index = [1]
    path_list = readFileData(rootdir, label_index, solvent)
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        left = len(file_list) - botom_index
        right = len(file_list) - 2
        for i in range(left, right):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # # current atom(include number)
            dict_solvent[temp_atom] = i - left
        break
    return dict_solvent


#
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


# def read_Speed(path):
#     return np.load(path, allow_pickle=True).item()

total_labels_ACE = readFileResult(args.result_src_ACE)
total_labels_NMF = readFileResult(args.result_src_NMF)
total_labels_wat = readFileResult(args.result_src_wat)
total_labels_meth = readFileResult(args.result_src_meth)
total_labels_DMF = readFileResult(args.result_src_DMF)


# total_labels_speed_ACE = read_Speed("./solute_solvent_speed_list_ACE_for_all_system.npy")
# total_labels_speed_NMF = read_Speed("./solute_solvent_speed_list_NMF_for_all_system.npy")
# total_labels_speed_wat = read_Speed("./solute_solvent_speed_list_wat_for_all_system.npy")
# total_labels_speed_meth = read_Speed("./solute_solvent_speed_list_meth_for_all_system.npy")


def readFileData(root_dir, lable_index, solvent):
    path_list = []
    for i in range(len(lable_index)):
        j = lable_index[i]
        temp_path = solvent + str(j) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb"))
    return path_list


def get_solute_adj(dict_solute):
    solute_adj = []
    bonds = 'S1-O3', 'S1-O4', 'S1-F1', 'S1-O1', 'O1-C3', 'C3-C4', 'C4-H3', 'C4-C5', 'C5-H4', 'C5-C6', 'C6-C1', 'C1-H1', 'C1-C2', 'C2-H2', 'C2-C3', 'C6-C19', 'C19-C14', 'C14-C13', 'C13-H9', 'C13-C18', 'C18-H12', 'C18-C17', 'C17-C16', 'C16-H11', 'C16-C15', 'C15-H10', 'C15-C14', 'C17-O2', 'O2-S2', 'S2-O5', 'S2-O6', 'S2-F2', 'C19-H13', 'C19-C10', 'C10-C11', 'C11-H7', 'C11-C12', 'C12-H8', 'C12-C7', 'C7-C8', 'C8-H5', 'C8-C9', 'C9-H6', 'C9-C10', 'C7-O25', 'O25-S8', 'S8-O26', 'S8-O27', 'S8-O22', 'O22-C64', 'C64-C69', 'C69-H47', 'C69-C68', 'C68-H46', 'C68-C67', 'C67-C66', 'C66-H45', 'C66-C65', 'C65-H44', 'C65-C64', 'C67-C76', 'C76-H52', 'C76-C71', 'C71-C70', 'C70-H48', 'C70-C75', 'C75-H51', 'C75-C74', 'C74-C73', 'C73-H50', 'C73-C72', 'C72-H49', 'C72-C71', 'C74-O24', 'O24-S9', 'S9-O29', 'S9-O30', 'S9-O28', 'O28-C26', 'C26-C27', 'C27-H18', 'C27-C28', 'C28-H19', 'C28-C29', 'C29-C30', 'C30-H20', 'C30-C31', 'C31-H21', 'C31-C26', 'C29-C38', 'C38-H26', 'C38-C25', 'C25-C20', 'C20-H14', 'C20-C21', 'C21-H15', 'C21-C22', 'C22-C23', 'C23-H16', 'C23-C24', 'C24-H17', 'C24-C25', 'C22-O7', 'O7-S3', 'S3-O9', 'S3-O10', 'S3-F3', 'C38-C33', 'C33-C32', 'C32-H22', 'C32-C37', 'C37-H25', 'C37-C36', 'C36-C35', 'C35-H24', 'C35-C34', 'C34-H23', 'C34-C33', 'C36-O8', 'O8-S4', 'S4-O11', 'S4-O12', 'S4-F4', 'C76-C63', 'C63-C62', 'C62-H43', 'C62-C61', 'C61-H42', 'C61-C60', 'C60-C59', 'C59-H41', 'C59-C58', 'C58-H40', 'C58-C63', 'C60-O23', 'O23-S7', 'S7-O20', 'S7-O21', 'S7-O17', 'O17-C55', 'C55-C56', 'C56-H38', 'C56-C51', 'C51-H35', 'C51-C52', 'C52-C53', 'C53-H36', 'C53-C54', 'C54-C55', 'C54-H37', 'C52-C57', 'C57-H39', 'C57-C48', 'C48-C47', 'C47-H32', 'C47-C46', 'C46-H31', 'C46-C45', 'C45-C50', 'C50-H34', 'C50-C49', 'C49-H33', 'C49-C48', 'C45-O13', 'O13-S5', 'S5-O14', 'S5-O15', 'S5-F5', 'C57-C44', 'C44-C43', 'C43-H30', 'C43-C42', 'C42-H29', 'C42-C41', 'C41-C40', 'C40-H28', 'C40-C39', 'C39-H27', 'C39-C44', 'C41-O16', 'O16-S6', 'S6-O18', 'S6-O19', 'S6-F6'
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(12 * args.batchsize):
            left_atom_value = dict_solute[left_atom] + 173 * i
            right_atom_value = dict_solute[right_atom] + 173 * i
            solute_adj.append((left_atom_value, right_atom_value))
            solute_adj.append((right_atom_value, left_atom_value))
    return torch.tensor(solute_adj).T


def get_solvent_adj(solvent, dict_solvent):
    if (solvent == "ACE"):
        atom_nums = 6
        atom_g = 1490
        bonds = 'C1-H1', 'C1-H2', 'C1-H3', 'C1-C2', 'C2-N1'
    elif (solvent == "NMF"):
        atom_nums = 9
        atom_g = 1350
        bonds = 'C1-O1', 'C1-H1', 'C1-N1', 'N1-H2', 'C2-N1', 'C2-H3', 'C2-H4', 'C2-H5'
    elif (solvent == "wat"):
        atom_nums = 3
        atom_g = 4928
        bonds = 'O-H1', 'O-H2'
    elif (solvent == "meth"):
        atom_nums = 15
        atom_g = 1089
        bonds = 'C7-H6', 'C7-H7', 'C7-H8', 'C7-C2', 'C2-C1', 'C1-H1', 'C1-C6', 'C6-H5', 'C6-C5', 'C5-H4', 'C5-C4', 'C4-H3', 'C4-C3', 'C3-H2', 'C3-C2'
    elif (solvent == "DMF"):
        atom_nums = 12
        atom_g = 1162
        bonds = 'C1-H1', 'C1-O1', 'C1-N1', 'N1-C2', 'C2-H2', 'C2-H3', 'C2-H4', 'N1-C3', 'C3-H5', 'C3-H6', 'C3-H7'
    solvent_adj = []
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(atom_g * args.batchsize):
            left_atom_value = dict_solvent[left_atom] + atom_nums * i
            right_atom_value = dict_solvent[right_atom] + atom_nums * i
            solvent_adj.append((left_atom_value, right_atom_value))
            solvent_adj.append((right_atom_value, left_atom_value))
    # A_loop = torch.eye(atom_nums * atom_g, atom_nums * atom_g)
    # solvent_adj = solvent_adj + A_loop
    return torch.tensor(solvent_adj).T


class test_trainset(Dataset):

    def __init__(self, phase, lable_index, solvent_test):
        path_list_test_solvent = readFileData(solvent_test["root_dir"], lable_index, solvent_test["solvent"])  #
        # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274
        self.solute_list_test, self.solvent_list_test, self.lables_list_test = self.__read_data__(
            path_list_test_solvent,
            lable_index, solvent_test["solvent"])

    def __read_data__(self, path_list, lable_index, solvent):
        solute_list = []
        solvent_list = []
        lables_list = []
        atom_to_feature = []
        total_labels = []
        if (solvent == "ACE"):
            atom_to_feature = ACE_atom_to_feature
            total_labels = total_labels_ACE
        elif (solvent == "NMF"):
            atom_to_feature = NMF_atom_to_feature
            total_labels = total_labels_NMF
        elif (solvent == "wat"):
            atom_to_feature = wat_atom_to_feature
            total_labels = total_labels_wat
        elif (solvent == "meth"):
            atom_to_feature = meth_atom_to_feature
            total_labels = total_labels_meth
        elif (solvent == "DMF"):
            atom_to_feature = DMF_atom_to_feature
            total_labels = total_labels_DMF
        #
        current_data = -1
        for path in path_list:
            current_data = current_data + 1
            if (current_data % 100 == 9):
                print("path=", path, "current_data=", current_data, ",all_data=", len(path_list), "")
            file = open(path, "r")
            file_list = file.readlines()
            temp_result_list = []
            position_list = []
            bottom = len(file_list) - 2

            for i in range(5, 2081):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  #
                # single_temp_atom = file_list[i][13]
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # print("solute_addtional_feature[temp_atom]=",solute_addtional_feature[temp_atom].shape)
                temp_result_list.append(solute_addtional_feature[temp_atom])
                position_list.append(x)
                position_list.append(y)
                position_list.append(z)

            for i in range(2081, bottom):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  #
                # single_temp_atom = file_list[i][13]
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # print("solute_addtional_feature[temp_atom]=",solute_addtional_feature[temp_atom].shape)
                temp_result_list.append(atom_to_feature[temp_atom])
                position_list.append(x)
                position_list.append(y)
                position_list.append(z)

            temp = torch.stack(temp_result_list)
            position_tensor = torch.tensor(position_list).reshape(-1, 3)

            # solute_speed = solute_speed_list[lable_index[current_data]]  # solute_speed_feature
            # solute_speed = solute_speed.reshape(-1, 1)
            # solvent_speed = solvent_speed_list[lable_index[current_data]]  # solvent_speed_feature
            # solvent_speed = solvent_speed.reshape(-1, 1)
            # speed = torch.cat((solute_speed, solvent_speed), 0)
            temp = torch.cat((temp, position_tensor), 1)
            # temp = torch.cat((temp, speed), 1)
            solute_data = temp[:173 * 12, :]
            solvent_data = temp[173 * 12:, :]
            solute_list.append(solute_data)  #
            solvent_list.append(solvent_data)  #
        for i in range(len(lable_index)):
            lables_list.append(total_labels[lable_index[i]])  #
        return solute_list, solvent_list, lables_list

    def __getitem__(self, index):
        solute_data_test = self.solute_list_test[index]
        solvent_data_test = self.solvent_list_test[index]
        lable_test = self.lables_list_test[index]
        return solute_data_test, solvent_data_test, lable_test

    def __len__(self):
        return len(self.solute_list_test)


class trainset(Dataset):

    def __init__(self, zero_solvent, phase, one_solvent, lable_index, two_solvent, three_solvent):

        path_list_zero_solvent = readFileData(zero_solvent["root_dir"], lable_index, zero_solvent["solvent"])  #
        path_list_one_solvent = readFileData(one_solvent["root_dir"], lable_index, one_solvent["solvent"])  #
        path_list_two_solvent = readFileData(two_solvent["root_dir"], lable_index, two_solvent["solvent"])  #
        path_list_three_solvent = readFileData(three_solvent["root_dir"], lable_index, three_solvent["solvent"])
        # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274

        self.solute_list_zero_solute, self.solvent_list_zero_solvent, self.lables_list_zero_solvent = self.__read_data__(
            path_list_zero_solvent,
            lable_index, zero_solvent["solvent"])
        self.solute_list_one_solute, self.solvent_list_one_solvent, self.lables_list_one_solvent = self.__read_data__(
            path_list_one_solvent,
            lable_index, one_solvent["solvent"])
        self.solute_list_two_solute, self.solvent_list_two_solvent, self.lables_list_two_solvent = self.__read_data__(
            path_list_two_solvent,
            lable_index, two_solvent["solvent"])
        self.solute_list_three_solute, self.solvent_list_three_solvent, self.lables_list_three_solvent = self.__read_data__(
            path_list_three_solvent,
            lable_index, three_solvent["solvent"])

    def __read_data__(self, path_list, lable_index, solvent):
        solute_list = []
        solvent_list = []
        lables_list = []
        atom_to_feature = []
        total_labels = []
        if (solvent == "ACE"):
            atom_to_feature = ACE_atom_to_feature
            total_labels = total_labels_ACE
        elif (solvent == "NMF"):
            atom_to_feature = NMF_atom_to_feature
            total_labels = total_labels_NMF
        elif (solvent == "wat"):
            atom_to_feature = wat_atom_to_feature
            total_labels = total_labels_wat
        elif (solvent == "meth"):
            atom_to_feature = meth_atom_to_feature
            total_labels = total_labels_meth
        elif (solvent == "DMF"):
            atom_to_feature = DMF_atom_to_feature
            total_labels = total_labels_DMF
        #
        current_data = -1
        for path in path_list:
            current_data = current_data + 1
            if (current_data % 1000 == 9):
                print("path=", path, "current_data=", current_data, ",all_data=", len(path_list), "")
            file = open(path, "r")
            file_list = file.readlines()
            temp_result_list = []
            position_list = []
            bottom = len(file_list) - 2

            for i in range(5, 2081):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  #
                # single_temp_atom = file_list[i][13]
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # print("solute_addtional_feature[temp_atom]=",solute_addtional_feature[temp_atom].shape)
                temp_result_list.append(solute_addtional_feature[temp_atom])  #
                position_list.append(x)
                position_list.append(y)
                position_list.append(z)

            for i in range(2081, bottom):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  #
                # single_temp_atom = file_list[i][13]
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # print("solute_addtional_feature[temp_atom]=",solute_addtional_feature[temp_atom].shape)
                temp_result_list.append(atom_to_feature[temp_atom])  #
                position_list.append(x)
                position_list.append(y)
                position_list.append(z)

            temp = torch.stack(temp_result_list)
            position_tensor = torch.tensor(position_list).reshape(-1, 3)
            temp = torch.cat((temp, position_tensor), 1)
            # temp = torch.cat((temp, speed), 1)
            solute_data = temp[:173 * 12, :]
            solvent_data = temp[173 * 12:, :]
            solute_list.append(solute_data)  #
            solvent_list.append(solvent_data)  #
        for i in range(len(lable_index)):
            lables_list.append(total_labels[lable_index[i]])  #
        return solute_list, solvent_list, lables_list

    def __getitem__(self, index):
        solute_data_zero = self.solute_list_zero_solute[index]
        solvent_data_zero = self.solvent_list_zero_solvent[index]
        # print("self.lables_list_ACE[index]=", self.lables_list_ACE[index])
        lable_zero = self.lables_list_zero_solvent[index]

        solute_data_one = self.solute_list_one_solute[index]
        solvent_data_one = self.solvent_list_one_solvent[index]
        lable_one = self.lables_list_one_solvent[index]

        solute_data_two = self.solute_list_two_solute[index]
        solvent_data_two = self.solvent_list_two_solvent[index]
        lable_two = self.lables_list_two_solvent[index]

        solute_data_three = self.solute_list_three_solute[index]
        solvent_data_three = self.solvent_list_three_solvent[index]
        lable_three = self.lables_list_three_solvent[index]

        return solute_data_zero, solvent_data_zero, lable_zero, solute_data_one, solvent_data_one, lable_one, solute_data_two, solvent_data_two, lable_two, solute_data_three, solvent_data_three, lable_three

    def __len__(self):
        return len(self.solute_list_zero_solute)


def load_data(zero_solvent, phase, one_solvent, lable_index, two_solvent, three_solvent):
    data = trainset(zero_solvent, phase, one_solvent, lable_index, two_solvent, three_solvent)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batchsize, shuffle=True, drop_last=True)
    return loader


def load_test_data(phase, label_index, solvent_test):
    data = test_trainset(phase, label_index, solvent_test)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batchsize, shuffle=True, drop_last=True)
    return loader


def val(net, val_loader, solute, zero_solvent, one_solvent, two_solvent, three_solvent):
    net.eval()
    SUMzero1 = 0
    SUMzero2 = 0
    SUMzero3 = 0
    SUMzero4 = 0
    SUMzero5 = 0

    SUMone1 = 0
    SUMone2 = 0
    SUMone3 = 0
    SUMone4 = 0
    SUMone5 = 0

    SUMtwo1 = 0
    SUMtwo2 = 0
    SUMtwo3 = 0
    SUMtwo4 = 0
    SUMtwo5 = 0

    SUMthree1 = 0
    SUMthree2 = 0
    SUMthree3 = 0
    SUMthree4 = 0
    SUMthree5 = 0
    for i_batch, batch_data in tqdm(enumerate(val_loader)):
        labels, solute_data_zero, solute_data_one, solute_data_two, solute_data_three, solvent_data_zero, solvent_data_one, solvent_data_two, solvent_data_three = para_to_device(
            batch_data)
        with torch.set_grad_enabled(False):  #
            outputs = net(solute_data_zero, solvent_data_zero, solute, zero_solvent,
                          solute_data_one, solvent_data_one, one_solvent,
                          solute_data_two, solvent_data_two, two_solvent,
                          solute_data_three, solvent_data_three, three_solvent).squeeze(-1)
        deviation = abs((outputs - labels) / labels)
        #         print("outputs=", outputs)
        #         print("labels=", labels)
        #         print("deviation=", deviation)
        for i in range(deviation.shape[0]):
            if (i < args.batchsize):
                if (deviation[i] < 0.01):
                    SUMzero1 = SUMzero1 + 1
                if (deviation[i] < 0.02):
                    SUMzero2 = SUMzero2 + 1
                if (deviation[i] < 0.03):
                    SUMzero3 = SUMzero3 + 1
                if (deviation[i] < 0.04):
                    SUMzero4 = SUMzero4 + 1
                if (deviation[i] < 0.05):
                    SUMzero5 = SUMzero5 + 1
            elif (i >= args.batchsize and i < 2 * args.batchsize):
                if (deviation[i] < 0.01):
                    SUMone1 = SUMone1 + 1
                if (deviation[i] < 0.02):
                    SUMone2 = SUMone2 + 1
                if (deviation[i] < 0.03):
                    SUMone3 = SUMone3 + 1
                if (deviation[i] < 0.04):
                    SUMone4 = SUMone4 + 1
                if (deviation[i] < 0.05):
                    SUMone5 = SUMone5 + 1
            elif (i >= 2 * args.batchsize and i < 3 * args.batchsize):
                if (deviation[i] < 0.01):
                    SUMtwo1 = SUMtwo1 + 1
                if (deviation[i] < 0.02):
                    SUMtwo2 = SUMtwo2 + 1
                if (deviation[i] < 0.03):
                    SUMtwo3 = SUMtwo3 + 1
                if (deviation[i] < 0.04):
                    SUMtwo4 = SUMtwo4 + 1
                if (deviation[i] < 0.05):
                    SUMtwo5 = SUMtwo5 + 1
            elif (i >= 3 * args.batchsize):
                if (deviation[i] < 0.01):
                    SUMthree1 = SUMthree1 + 1
                if (deviation[i] < 0.02):
                    SUMthree2 = SUMthree2 + 1
                if (deviation[i] < 0.03):
                    SUMthree3 = SUMthree3 + 1
                if (deviation[i] < 0.04):
                    SUMthree4 = SUMthree4 + 1
                if (deviation[i] < 0.05):
                    SUMthree5 = SUMthree5 + 1
    # print("SUMACE1 = ", SUMACE1, "SUMACE2 = ", SUMACE2, "SUMACE3 = ", SUMACE3, "SUMACE4 = ", SUMACE4, "SUMACE5 = ",
    #       SUMACE5)
    # print("SUMNMF1 = ", SUMNMF1, "SUMNMF2 = ", SUMNMF2, "SUMNMF3 = ", SUMNMF3, "SUMNMF4 = ", SUMNMF4, "SUMNMF5 = ",
    #       SUMNMF5)
    # print("SUMwat1 = ", SUMwat1, "SUMwat2 = ", SUMwat2, "SUMwat3 = ", SUMwat3, "SUMwat4 = ", SUMwat4,
    #       "SUMwat5 = ", SUMwat5)
    zero_accuracy = (SUMzero5) / (len(val_loader) * args.batchsize)
    one_accuracy = (SUMone5) / (len(val_loader) * args.batchsize)
    two_accuracy = (SUMtwo5) / (len(val_loader) * args.batchsize)
    three_accuracy = (SUMthree5) / (len(val_loader) * args.batchsize)

    print("tatol", zero_accuracy * 100, "% the_zero_solvent in 5%;", "tatol",
          one_accuracy * 100, "% the_firt_solvent in 5%;", "tatol",
          two_accuracy * 100, "% the_second_solvent in 5%", "tatol",
          three_accuracy * 100, "% the_third_solvent in 5%")
    return zero_accuracy, one_accuracy, two_accuracy, three_accuracy


def val_another_solvent(test_model, test_loader, solute, solvent_test, PATH1, bestSUM_solvent_in_5,
                        bestSUM_solvent_in_10, best_PATH5,
                        best_PATH10, current_iter, zero_accuracy, one_accuracy, two_accuracy, three_accuracy,
                        current_epoch, batch_num, schedular):
    test_model.eval()
    SUMtest1 = 0
    SUMtest2 = 0
    SUMtest3 = 0
    SUMtest4 = 0
    SUMtest5 = 0
    SUMtest6 = 0
    SUMtest7 = 0
    SUMtest8 = 0
    SUMtest9 = 0
    SUMtest10 = 0

    for i_batch, batch_data in tqdm(enumerate(test_loader)):
        solute_data_test, solvent_data_test, label_test = batch_data
        labels = label_test.to(DEVICE)
        solute_data_test = solute_data_test.to(DEVICE)
        solvent_data_test = solvent_data_test.to(DEVICE)
        with torch.set_grad_enabled(False):  #
            outputs = test_model(solute, solute_data_test, solvent_data_test, solvent_test).squeeze(-1)
        deviation = abs((outputs - labels) / labels)
        if (i_batch % 100 == 0):
            print("outputs=", outputs)
            print("labels=", labels)
            print("deviation=", deviation)
        for i in range(deviation.shape[0]):
            if (i < args.batchsize):
                if (deviation[i] < 0.01):
                    SUMtest1 = SUMtest1 + 1
                if (deviation[i] < 0.02):
                    SUMtest2 = SUMtest2 + 1
                if (deviation[i] < 0.03):
                    SUMtest3 = SUMtest3 + 1
                if (deviation[i] < 0.04):
                    SUMtest4 = SUMtest4 + 1
                if (deviation[i] < 0.05):
                    SUMtest5 = SUMtest5 + 1
                if (deviation[i] < 0.06):
                    SUMtest6 = SUMtest6 + 1
                if (deviation[i] < 0.07):
                    SUMtest7 = SUMtest7 + 1
                if (deviation[i] < 0.08):
                    SUMtest8 = SUMtest8 + 1
                if (deviation[i] < 0.09):
                    SUMtest9 = SUMtest9 + 1
                if (deviation[i] < 0.1):
                    SUMtest10 = SUMtest10 + 1
    if (SUMtest10 > bestSUM_solvent_in_10):
        bestSUM_solvent_in_10 = SUMtest10
        best_PATH10 = PATH1 + "-MySuperGATTest4_net_in_10.pth.tar"
    if (SUMtest5 > bestSUM_solvent_in_5):
        bestSUM_solvent_in_5 = SUMtest5
        best_PATH5 = PATH1 + "-MySuperGATTest4_net_in_5.pth.tar"
    # print("SUMmeth1 = ", SUMmeth1, "SUMmeth2 = ", SUMmeth2, "SUMmeth3 = ", SUMmeth3, "SUMmeth4 = ", SUMmeth4,
    #       "SUMmeth5 = ", SUMmeth5)
    # print("SUMmeth6 = ", SUMmeth6, "SUMmeth7 = ", SUMmeth7, "SUMmeth8 = ", SUMmeth8, "SUMmeth9 = ", SUMmeth9,
    #       "SUMmeth10 = ", SUMmeth10)
    test_accuracy = (SUMtest5) / (len(test_loader) * args.batchsize)
    schedular.step(metrics=test_accuracy)
    writer.add_scalars('Accuracy',
                       {"zero_acc": zero_accuracy, "one_acc": one_accuracy, "two_acc": two_accuracy,
                        "three_acc": three_accuracy},
                       current_iter + current_epoch * batch_num)
    writer.add_scalars('Accuracy',
                       {"test_acc": test_accuracy},
                       current_iter + current_epoch * batch_num)
    print("tatol", (SUMtest5 * 100) / (len(test_loader) * args.batchsize), "% test_solvent in 5%;", "tatol",
          (SUMtest10 * 100) / (len(test_loader) * args.batchsize), "% test_solvent in 10%", ";current_iter=",
          current_iter)
    return best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10


def build_model(checkpoint_PATH):
    net = MyNewGNN(nfeat=args.nfeat, nhid=args.nhid, nclass=args.nclass, dropout=args.dropout, DEVICE=DEVICE,
                   batch_size=args.batchsize)
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    schedular_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience,
                                                             verbose=True, eps=1e-8)
    # schedular_r = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    schedular = GradualWarmupScheduler(optimizer, multiplier=1000, total_epoch=args.warmup_step,
                                       after_scheduler=schedular_r)

    criterion = torch.nn.MSELoss()
    test_model = MyValModel(nfeat=args.nfeat, nhid=args.nhid, nclass=args.nclass, dropout=args.dropout, DEVICE=DEVICE,
                            batch_size=args.batchsize)
    test_model = test_model.to(DEVICE)
    if (checkpoint_PATH != None):
        model_CKPT = torch.load(checkpoint_PATH)
        net.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        print("optimizer=", optimizer)
    return net, optimizer, criterion, test_model, schedular


def para_to_device(batch_data):
    solute_data_zero, solvent_data_zero, label_zero, solute_data_one, solvent_data_one, label_one, solute_data_two, solvent_data_two, label_two, solute_data_three, solvent_data_three, label_three = batch_data

    labels_zero = label_zero.to(DEVICE)
    labels_one = label_one.to(DEVICE)
    labels_two = label_two.to(DEVICE)
    labels_three = label_three.to(DEVICE)

    labels = torch.cat((labels_zero, labels_one), 0)
    labels = torch.cat((labels, labels_two), 0)
    labels = torch.cat((labels, labels_three), 0)
    solute_data_zero = solute_data_zero.to(DEVICE)
    solute_data_one = solute_data_one.to(DEVICE)
    solute_data_two = solute_data_two.to(DEVICE)
    solute_data_three = solute_data_three.to(DEVICE)
    solvent_data_zero = solvent_data_zero.to(DEVICE)
    solvent_data_one = solvent_data_one.to(DEVICE)
    solvent_data_two = solvent_data_two.to(DEVICE)
    solvent_data_three = solvent_data_three.to(DEVICE)
    return labels, solute_data_zero, solute_data_one, solute_data_two, solute_data_three, solvent_data_zero, solvent_data_one, solvent_data_two, solvent_data_three


def train(net, optimizer, criterion, test_model, train_loader, solute, zero_solvent, one_solvent, two_solvent,
          three_solvent, val_loader, solvent_test, test_loader, best_PATH5, best_PATH10, bestSUM_solvent_in_5,
          bestSUM_solvent_in_10, current_iter, current_epoch, batch_num, schedular):
    loss_list = []
    # tf_flag = 0
    net.train()
    total_loss = 0
    for i_batch, batch_data in tqdm(enumerate(train_loader)):
        labels, solute_data_zero, solute_data_one, solute_data_two, solute_data_three, solvent_data_zero, solvent_data_one, solvent_data_two, solvent_data_three = para_to_device(
            batch_data)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):  #
            outputs = net(solute_data_zero, solvent_data_zero, solute, zero_solvent,
                          solute_data_one, solvent_data_one, one_solvent,
                          solute_data_two, solvent_data_two, two_solvent,
                          solute_data_three, solvent_data_three, three_solvent).squeeze(-1)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if (i_batch % 1000 == 9):
        # print("outputs=", outputs)
        # print("labels=", labels)
        # print("loss=", loss)

    print("current_lr==================================================================================",
          optimizer.state_dict()['param_groups'][0]['lr'])
    print("schedular.last_epoch====================", schedular.last_epoch)
    writer.add_scalar('current_lr', optimizer.state_dict()['param_groups'][0]['lr'],
                      current_iter + current_epoch * batch_num)
    writer.add_scalar('total loss', total_loss / len(train_loader), current_iter + current_epoch * batch_num)
    loss_list.append(total_loss / len(train_loader))
    PATH1 = "./check_point/MySuperGATTest4/" + str(current_epoch) + "/" + str(current_iter)
    # torch.save(net.state_dict(), PATH1)
    if (current_iter % 20 == 0):
        print("save model, current_iter=", current_iter)
        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                   "./check_point/MySuperGATTest4/" + str(current_epoch) + "/" + str(
                       current_iter) + "-MySuperGATTest4_net.pth.tar")
    zero_accuracy, one_accuracy, two_accuracy, three_accuracy = val(net, val_loader, solute, zero_solvent, one_solvent,
                                                                    two_solvent, three_solvent)

    test_model.load_state_dict(net.state_dict())
    best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10 = val_another_solvent(test_model, test_loader,
                                                                                               solute, solvent_test,
                                                                                               PATH1,
                                                                                               bestSUM_solvent_in_5,
                                                                                               bestSUM_solvent_in_10,
                                                                                               best_PATH5,
                                                                                               best_PATH10,
                                                                                               current_iter,
                                                                                               zero_accuracy,
                                                                                               one_accuracy,
                                                                                               two_accuracy,
                                                                                               three_accuracy,
                                                                                               current_epoch, batch_num,
                                                                                               schedular)

    if ((best_PATH5.endswith(".pth.tar")) and (os.path.exists(best_PATH5) == False)):
        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, best_PATH5)
    if ((best_PATH10.endswith(".pth.tar")) and (os.path.exists(best_PATH10) == False)):
        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, best_PATH10)
    print("best_PATH5=", best_PATH5, "best_PATH10=", best_PATH10)
    return best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10


def LOAD(train_lable_index, val_lable_index, test_list, zero_solvent, one_solvent, two_solvent, three_solvent,
         solvent_test):
    train_loader = load_data(zero_solvent, "train", one_solvent, train_lable_index, two_solvent,
                             three_solvent)  #
    val_loader = load_data(zero_solvent, "val", one_solvent, val_lable_index, two_solvent,
                           three_solvent)  #
    test_loader = load_test_data("test", test_list, solvent_test)  #
    return train_loader, val_loader, test_loader


def TRAIN(net, optimizer, criterion, test_model, train_loader, solute, zero_solvent, one_solvent, two_solvent,
          three_solvent, val_loader, solvent_test, test_loader, best_PATH5, best_PATH10, bestSUM_solvent_in_5,
          bestSUM_solvent_in_10, current_iter, current_epoch, batch_num, schedular):
    best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10 = train(net, optimizer, criterion, test_model,
                                                                                 train_loader, solute, zero_solvent,
                                                                                 one_solvent, two_solvent,
                                                                                 three_solvent, val_loader,
                                                                                 solvent_test, test_loader, best_PATH5,
                                                                                 best_PATH10, bestSUM_solvent_in_5,
                                                                                 bestSUM_solvent_in_10, current_iter,
                                                                                 current_epoch, batch_num, schedular)
    return best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10


def split_train_and_test_solvent(test_index):
    dict_index_to_solvent = {0: "ACE", 1: "NMF", 2: "DMF", 3: "wat", 4: "meth"}
    # dict_solvent_to_solventAtomNums = {"ACE": 8940, "NMF": 12150, "DMF": 13944, "wat": 14784, "meth": 16335}
    dict_solvent_to_rootdir = {"ACE": args.data_src_ACE, "NMF": args.data_src_NMF, "DMF": args.data_src_DMF,
                               "wat": args.data_src_wat, "meth": args.data_src_meth}
    # get adj
    dict_solute = get_solute_position(args.data_src_ACE)
    dict_solvent_NMF = get_solvent_position(args.data_src_NMF, "NMF")
    dict_solvent_ACE = get_solvent_position(args.data_src_ACE, "ACE")
    dict_solvent_wat = get_solvent_position(args.data_src_wat, "wat")
    dict_solvent_meth = get_solvent_position(args.data_src_meth, "meth")
    dict_solvent_DMF = get_solvent_position(args.data_src_DMF, "DMF")

    dict_solvent_to_position = {"ACE": dict_solvent_ACE, "NMF": dict_solvent_NMF, "wat": dict_solvent_wat,
                                "DMF": dict_solvent_DMF, "meth": dict_solvent_meth}
    dict_solvent_to_embedding = {"ACE": torch.from_numpy(np.load("./ACE_feature.npy", allow_pickle=True)),
                                 "NMF": torch.from_numpy(np.load("./NMF_feature.npy", allow_pickle=True)),
                                 "DMF": torch.from_numpy(np.load("./DMF_feature.npy", allow_pickle=True)),
                                 "wat": torch.from_numpy(np.load("./wat_feature.npy", allow_pickle=True)),
                                 "meth": torch.from_numpy(np.load("./meth_feature.npy", allow_pickle=True))}
    solute = {"solute_adj": get_solute_adj(dict_solute),
              "solute_to_embedding": torch.from_numpy(np.load("./solute_feature.npy", allow_pickle=True))}
    # solute_adj = get_solute_adj()
    # solute_adj = torch.tensor(solute_adj).T
    # solvent_adj_NMF = get_solvent_adj("NMF", 12150)
    # solvent_adj_NMF = torch.tensor(solvent_adj_NMF).T
    # solvent_adj_ACE = get_solvent_adj("ACE", 8940)
    # solvent_adj_ACE = torch.tensor(solvent_adj_ACE).T
    # solvent_adj_wat = get_solvent_adj("wat", 14784)
    # solvent_adj_wat = torch.tensor(solvent_adj_wat).T
    # solvent_adj_meth = get_solvent_adj("meth", 16335)
    # solvent_adj_meth = torch.tensor(solvent_adj_meth).T
    # solvent_adj_DMF = get_solvent_adj("DMF", 13944)
    # solvent_adj_DMF = torch.tensor(solvent_adj_DMF).T

    # solvent_
    solvent_list = []
    temp_zero_solvent = {}
    temp_one_solvent = {}
    temp_two_solvent = {}
    temp_three_solvent = {}
    temp_four_solvent = {}
    for i in range(args.total_dataset):
        if (i == 0):
            temp_zero_solvent["solvent"] = dict_index_to_solvent[i]
            temp_zero_solvent["solvent_adj"] = get_solvent_adj(temp_zero_solvent["solvent"],
                                                               dict_solvent_to_position[temp_zero_solvent["solvent"]])
            temp_zero_solvent["root_dir"] = dict_solvent_to_rootdir[temp_zero_solvent["solvent"]]
            temp_zero_solvent["smile_to_vector"] = dict_solvent_to_embedding[temp_zero_solvent["solvent"]]
            solvent_list.append(temp_zero_solvent)
        elif (i == 1):
            temp_one_solvent["solvent"] = dict_index_to_solvent[i]
            temp_one_solvent["solvent_adj"] = get_solvent_adj(temp_one_solvent["solvent"],
                                                              dict_solvent_to_position[temp_one_solvent["solvent"]])
            temp_one_solvent["root_dir"] = dict_solvent_to_rootdir[temp_one_solvent["solvent"]]
            temp_one_solvent["smile_to_vector"] = dict_solvent_to_embedding[temp_one_solvent["solvent"]]
            solvent_list.append(temp_one_solvent)
        elif (i == 2):
            temp_two_solvent["solvent"] = dict_index_to_solvent[i]
            temp_two_solvent["solvent_adj"] = get_solvent_adj(temp_two_solvent["solvent"],
                                                              dict_solvent_to_position[temp_two_solvent["solvent"]])
            temp_two_solvent["root_dir"] = dict_solvent_to_rootdir[temp_two_solvent["solvent"]]
            temp_two_solvent["smile_to_vector"] = dict_solvent_to_embedding[temp_two_solvent["solvent"]]
            solvent_list.append(temp_two_solvent)
        elif (i == 3):
            temp_three_solvent["solvent"] = dict_index_to_solvent[i]
            temp_three_solvent["solvent_adj"] = get_solvent_adj(temp_three_solvent["solvent"],
                                                                dict_solvent_to_position[temp_three_solvent["solvent"]])
            temp_three_solvent["root_dir"] = dict_solvent_to_rootdir[temp_three_solvent["solvent"]]
            temp_three_solvent["smile_to_vector"] = dict_solvent_to_embedding[temp_three_solvent["solvent"]]
            solvent_list.append(temp_three_solvent)
        elif (i == 4):
            temp_four_solvent["solvent"] = dict_index_to_solvent[i]
            temp_four_solvent["solvent_adj"] = get_solvent_adj(temp_four_solvent["solvent"],
                                                               dict_solvent_to_position[temp_four_solvent["solvent"]])
            temp_four_solvent["root_dir"] = dict_solvent_to_rootdir[temp_four_solvent["solvent"]]
            temp_four_solvent["smile_to_vector"] = dict_solvent_to_embedding[temp_four_solvent["solvent"]]
            solvent_list.append(temp_four_solvent)

    solvent_test = solvent_list[test_index]
    for j in range(test_index, args.total_dataset - 1):
        solvent_list[j] = solvent_list[j + 1]
    zero_solvent = solvent_list[0]
    one_solvent = solvent_list[1]
    two_solvent = solvent_list[2]
    three_solvent = solvent_list[3]

    solute["solute_adj"] = solute["solute_adj"].to(DEVICE)
    solute["solute_to_embedding"] = solute["solute_to_embedding"].to(DEVICE)

    zero_solvent["solvent_adj"] = zero_solvent["solvent_adj"].to(DEVICE)
    zero_solvent["smile_to_vector"] = zero_solvent["smile_to_vector"].to(DEVICE)

    one_solvent["solvent_adj"] = one_solvent["solvent_adj"].to(DEVICE)
    one_solvent["smile_to_vector"] = one_solvent["smile_to_vector"].to(DEVICE)

    two_solvent["solvent_adj"] = two_solvent["solvent_adj"].to(DEVICE)
    two_solvent["smile_to_vector"] = two_solvent["smile_to_vector"].to(DEVICE)

    three_solvent["solvent_adj"] = three_solvent["solvent_adj"].to(DEVICE)
    three_solvent["smile_to_vector"] = three_solvent["smile_to_vector"].to(DEVICE)

    solvent_test["solvent_adj"] = solvent_test["solvent_adj"].to(DEVICE)
    solvent_test["smile_to_vector"] = solvent_test["smile_to_vector"].to(DEVICE)

    return solute, zero_solvent, one_solvent, two_solvent, three_solvent, solvent_test


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    seed_torch(42)
    data_list = []
    print("--------------------------build_model----------------------------")
    net, optimizer, criterion, test_model, schedular = build_model(checkpoint_PATH=None)
    net.apply(weight_init)

    solute, zero_solvent, one_solvent, two_solvent, three_solvent, solvent_test = split_train_and_test_solvent(
        args.test_index)

    meth_atom_to_feature = np.load("./meth_atom_to_feature.npy", allow_pickle=True).item()
    wat_atom_to_feature = np.load("./wat_atom_to_feature.npy", allow_pickle=True).item()
    NMF_atom_to_feature = np.load("./NMF_atom_to_feature.npy", allow_pickle=True).item()
    ACE_atom_to_feature = np.load("./ACE_atom_to_feature.npy", allow_pickle=True).item()
    DMF_atom_to_feature = np.load("./DMF_atom_to_feature.npy", allow_pickle=True).item()
    solute_addtional_feature = np.load("./solute_atom_to_feature.npy", allow_pickle=True).item()

    for i in range(0, args.total_data):
        data_list.append(i)
    train_lable_index, val_lable_index = train_test_split(data_list, train_size=0.8, random_state=42)
    val_lable_index, test_lable_index = train_test_split(val_lable_index, train_size=0.5, random_state=42)
    batch = args.batchsize * 8
    val_batch = int(batch / 8)
    batch_num = int(len(train_lable_index) / batch)

    # i = 0
    for current_epoch in range(args.epoch):
        root_path = "./check_point/MySuperGATTest4/" + str(current_epoch)
        if (os.path.exists(root_path) == False):
            os.mkdir(root_path)
        best_PATH5 = ""
        best_PATH10 = ""
        bestSUM_solvent_in_5 = 0
        bestSUM_solvent_in_10 = 0
        for current_iter in range(batch_num):
            print("current_epoch=", current_epoch, "current_iter=", current_iter,
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            temp_train_lable_index = train_lable_index[current_iter * batch:(current_iter + 1) * batch]
            temp_val_lable_index = val_lable_index[current_iter * val_batch: int(current_iter + 1) * val_batch]
            test_list = temp_val_lable_index

            train_loader, val_loader, test_loader = LOAD(temp_train_lable_index, temp_val_lable_index, test_list,
                                                         zero_solvent, one_solvent, two_solvent, three_solvent,
                                                         solvent_test)

            best_PATH5, best_PATH10, bestSUM_solvent_in_5, bestSUM_solvent_in_10 = TRAIN(net, optimizer, criterion,
                                                                                         test_model, train_loader,
                                                                                         solute, zero_solvent,
                                                                                         one_solvent,
                                                                                         two_solvent, three_solvent,
                                                                                         val_loader,
                                                                                         solvent_test, test_loader,
                                                                                         best_PATH5, best_PATH10,
                                                                                         bestSUM_solvent_in_5,
                                                                                         bestSUM_solvent_in_10,
                                                                                         current_iter, current_epoch,
                                                                                         batch_num, schedular)