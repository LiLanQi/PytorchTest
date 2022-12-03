# encoding=utf-8
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os
from MyLastGCNTestmeth9 import MyValModel
from MyLastGCN9 import MyNewGCN
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 用ACE+NMF+Water进行模型训练，测试meth,当前正在测试溶质与溶剂间原子的距离
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
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--dropout', type=float, default=0)

DEVICE = torch.device('cuda:1')
args = parser.parse_args()
writer = SummaryWriter('MyLastGCNTest9')

dict_solute = {}
dict_solvent_ACE = {}
dict_solvent_NMF = {}
dict_solvent_wat = {}
dict_solvent_meth = {}

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.numpy()
    rowsum = np.array(mx.sum(1))  # 对每一个特征进行归一化
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
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
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
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
            dict_solvent[temp_atom] = i - left
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
    solute_adj = torch.zeros(173 * 12, 173 * 12)
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
        bonds = 'C7-H6', 'C7-H7', 'C7-H8', 'C7-C2', 'C2-C1', 'C1-H1', 'C1-C6', 'C6-H5', 'C6-C5', 'C5-H4', 'C5-C4', 'C4-H3', 'C4-C3', 'C3-H2', 'C3-C2'
        dict_solvent = dict_solvent_meth
    solvent_adj = torch.zeros(dim, dim)
    for connect_atom in bonds:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        for i in range(atom_g):
            left_atom_value = dict_solvent[left_atom] + atom_nums * i
            right_atom_value = dict_solvent[right_atom] + atom_nums * i
            solvent_adj[left_atom_value][right_atom_value] = 1
            solvent_adj[right_atom_value][left_atom_value] = 1
    A_loop = torch.eye(atom_nums * atom_g, atom_nums * atom_g)
    solvent_adj = solvent_adj + A_loop
    return solvent_adj

def minkowski_distance_p(x, y, p=2):
    """
    Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    # >>> from scipy.spatial import minkowski_distance_p
    # >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """
    Compute the L**p distance between two arrays.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    # >>> from scipy.spatial import minkowski_distance
    # >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


def distance_matrix(x, y, p=2, threshold=1000000):
    """
    Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.

    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.

    Examples
    --------
    # >>> from scipy.spatial import distance_matrix
    # >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    """

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    if m*n*k <= threshold:
        return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
    else:
        result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p)
        return result


class test_trainset(Dataset):

    def __init__(self, phase, lable_index, rootdir_meth):
        path_list_meth = readFileData(rootdir_meth, lable_index, "meth")  # 得到对应methx.pdb文件的list
        # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274
        self.solute_list_meth, self.solvent_list_meth, self.lables_list_meth, self.solute_meth_adj_list = self.__read_data__(path_list_meth, lable_index, "meth", solute_adj, solvent_adj_meth)

    def edge_ligand_pocket(self, dist_matrix, solute_size, theta=4, keep_pock=False):
        """
        Extract the edges between the ligand and protein-pocket.
        """
        pos = np.where(dist_matrix <= theta)
        # ligand_list, pocket_list = pos

        solute_list, solvent_list = pos
        solute_size = dist_matrix.shape[0]
        solute_solvent_adj = torch.zeros(dist_matrix.shape[0] + dist_matrix.shape[1],
                                         dist_matrix.shape[0] + dist_matrix.shape[1])
        for i, j in zip(solute_list, solvent_list):
            solute_solvent_adj[i][j + solute_size] = 1
            solute_solvent_adj[j + solute_size][i] = 1
        return solute_solvent_adj

    def cons_spatial_gragh(self, dist_matrix, theta=5):
        """
        Construct the spatial graph based on the cutoff theta.
        """
        pos = np.where((dist_matrix <= theta) & (dist_matrix != 0))
        src_list, dst_list = pos
        dist_list = dist_matrix[pos]
        edges = [(x, y) for x, y in zip(src_list, dst_list)]
        return edges

    def solvent_subgraph(self, node_map, edge_list):
        """
        Extract the subgraph of protein-pocket from the edge_list.
        """
        edge_l = []
        node_l = set()
        for x, y in edge_list:
            if x in node_map and y in node_map:
                x, y = node_map[x], node_map[y]
                edge_l.append((x, y))
                node_l.add(x)
                node_l.add(y)
        return edge_l

    def __cons_solute_solvent_graph_with_spatial_context__(self, solute_feature, solute_pos, solute_adj,
                                                           solvent_feature, solvent_pos, solvent_adj, add_fea=2,
                                                           theta=5, keep_pock=False, pocket_spatial=True):

        # inter-relation between solute and solvent
        solute_size = solute_feature.shape[0]
        dm = distance_matrix(solute_pos, solvent_pos)
        dist_list, solute_solvent_edge, node_map = self.edge_ligand_pocket(dm, solute_size, theta=theta,
                                                                           keep_pock=keep_pock)

        # construct ligand graph & pocket graph
        # lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
        # pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)

        # construct spatial context graph based on distance
        dm = distance_matrix(solute_pos, solute_pos)
        edges = self.cons_spatial_gragh(dm, theta=theta)
        if pocket_spatial:
            dm_solvent = distance_matrix(solvent_feature, solvent_feature)
            edges_solvent = self.cons_spatial_gragh(dm_solvent, theta=theta)
        solute_edge = edges
        solvent_edge = edges_solvent

        # map new pocket graph
        # solvent_size = len(node_map)
        # solvent_feature = solvent_feature[sorted(node_map.keys())]
        solvent_edge = self.solvent_subgraph(node_map, solvent_edge)
        # solvent_pos = solvent_pos[sorted(node_map.keys())]

        # construct ligand-pocket graph
        # size = solute_size + solvent_size

        # feas = np.vstack([solute_feature, solvent_feature])
        edges = solute_edge + solute_solvent_edge + solvent_edge
        # coords = np.vstack([solute_pos, solvent_pos])
        # edges = [(i, j) for i, j in edges]
        # print("size=",size)
        # print("feas=",feas)
        # print("edges=",edges)
        # print("coords=",coords)
        return edges

    def __read_data__(self, path_list, lable_index, solvent, solute_adj, solvent_adj):
        solute_list = []
        solvent_list = []
        lables_list = []
        atom_to_feature = []
        total_labels = []
        solute_solvent_adj_list = []
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
        # 对所有的ACE.pdb文件里的数据进行预处理
        current_data = 0
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
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
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
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
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
            temp = torch.cat((temp, position_tensor), 1)
            solute_data = temp[:173 * 12, :]
            solvent_data = temp[173 * 12:, :]
            solute_list.append(solute_data)  # 存入每个ACE文件的溶质图
            solvent_list.append(solvent_data)  # 存入每个ACE文件的溶剂图
            solute_position = position_tensor[:173 * 12, :]
            solvent_position = position_tensor[173 * 12:, :]
            dm = distance_matrix(solute_position, solvent_position)
            solute_solvent_adj = self.edge_ligand_pocket(dm, solute_data.shape[0])
            # edges= self.__cons_solute_solvent_graph_with_spatial_context__(solute_data[:,:-3], solute_position, solute_adj, solvent_data[:,:-3], solvent_position, solvent_adj)
            solute_solvent_adj_list.append(solute_solvent_adj)
        for i in range(len(lable_index)):
            lables_list.append(total_labels[lable_index[i]])  # 存入每个ACE文件的标签
        return solute_list, solvent_list, lables_list, solute_solvent_adj_list

    def __getitem__(self, index):
        solute_list_meth = self.solute_list_meth[index]
        solvent_list_meth = self.solvent_list_meth[index]
        lable_meth = self.lables_list_meth[index]
        solute_and_meth_adj = self.solute_meth_adj_list[index]

        return solute_list_meth, solvent_list_meth, lable_meth, solute_and_meth_adj

    def __len__(self):
        return len(self.solute_list_meth)

class trainset(Dataset):

    def __init__(self, rootdir_ACE, phase, rootdir_NMF,  lable_index, rootdir_wat):
        self.root = rootdir_ACE
        path_list_ACE = readFileData(rootdir_ACE, lable_index, "ACE")  # 得到对应ACEx.pdb文件的list
        path_list_NMF = readFileData(rootdir_NMF, lable_index, "NMF")  # 得到对应NMFx.pdb文件的list
        path_list_wat = readFileData(rootdir_wat, lable_index, "wat")  # 得到对应methx.pdb文件的list
        # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274

        self.solute_list_ACE, self.solvent_list_ACE, self.lables_list_ACE, self.solute_ACE_adj_list = self.__read_data__(path_list_ACE, lable_index, "ACE",solute_adj, solvent_adj_ACE)
        self.solute_list_NMF, self.solvent_list_NMF, self.lables_list_NMF, self.solute_NMF_adj_list = self.__read_data__(path_list_NMF, lable_index, "NMF",solute_adj, solvent_adj_NMF)
        self.solute_list_wat, self.solvent_list_wat, self.lables_list_wat, self.solute_wat_adj_list = self.__read_data__(path_list_wat, lable_index, "wat",solute_adj, solvent_adj_wat)

    def edge_ligand_pocket(self, dist_matrix, solute_size, theta=4, keep_pock=False):
        """
        Extract the edges between the ligand and protein-pocket.
        """
        pos = np.where(dist_matrix <= theta)
        # ligand_list, pocket_list = pos

        solute_list, solvent_list = pos
        solute_size = dist_matrix.shape[0]
        solute_solvent_adj = torch.zeros(dist_matrix.shape[0]+dist_matrix.shape[1], dist_matrix.shape[0]+dist_matrix.shape[1])
        for i,j in zip(solute_list, solvent_list):
            solute_solvent_adj[i][j+solute_size] = 1
            solute_solvent_adj[j+solute_size][i] = 1
        return solute_solvent_adj

    def cons_spatial_gragh(self, dist_matrix, theta=5):
        """
        Construct the spatial graph based on the cutoff theta.
        """
        pos = np.where((dist_matrix <= theta) & (dist_matrix != 0))
        src_list, dst_list = pos
        dist_list = dist_matrix[pos]
        edges = [(x, y) for x, y in zip(src_list, dst_list)]
        return edges

    def solvent_subgraph(self, node_map, edge_list):
        """
        Extract the subgraph of protein-pocket from the edge_list.
        """
        edge_l = []
        node_l = set()
        for x, y in edge_list:
            if x in node_map and y in node_map:
                x, y = node_map[x], node_map[y]
                edge_l.append((x, y))
                node_l.add(x)
                node_l.add(y)
        return edge_l

    def __cons_solute_solvent_graph_with_spatial_context__(self, solute_feature, solute_pos, solute_adj, solvent_feature, solvent_pos,solvent_adj, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True):

        # inter-relation between solute and solvent
        solute_size = solute_feature.shape[0]
        dm = distance_matrix(solute_pos, solvent_pos)
        dist_list, solute_solvent_edge, node_map = self.edge_ligand_pocket(dm, solute_size, theta=theta, keep_pock=keep_pock)

        # construct ligand graph & pocket graph
        # lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
        # pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)

        # construct spatial context graph based on distance
        dm = distance_matrix(solute_pos, solute_pos)
        edges = self.cons_spatial_gragh(dm, theta=theta)
        if pocket_spatial:
            dm_solvent = distance_matrix(solvent_feature, solvent_feature)
            edges_solvent = self.cons_spatial_gragh(dm_solvent, theta=theta)
        solute_edge = edges
        solvent_edge = edges_solvent

        # map new pocket graph
        # solvent_size = len(node_map)
        # solvent_feature = solvent_feature[sorted(node_map.keys())]
        solvent_edge = self.solvent_subgraph(node_map, solvent_edge)
        # solvent_pos = solvent_pos[sorted(node_map.keys())]

        # construct ligand-pocket graph
        # size = solute_size + solvent_size

        # feas = np.vstack([solute_feature, solvent_feature])
        edges = solute_edge + solute_solvent_edge + solvent_edge
        # coords = np.vstack([solute_pos, solvent_pos])
        # edges = [(i, j) for i, j in edges]
        # print("size=",size)
        # print("feas=",feas)
        # print("edges=",edges)
        # print("coords=",coords)
        return edges

    def __read_data__(self, path_list, lable_index, solvent, solute_adj, solvent_adj):
        solute_list = []
        solvent_list = []
        lables_list = []
        atom_to_feature = []
        total_labels = []
        solute_solvent_adj_list = []
        if(solvent == "ACE"):
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
        # 对所有的ACE.pdb文件里的数据进行预处理
        current_data = 0
        for path in path_list:
            current_data = current_data + 1
            # if (current_data % 1000 == 9):
            print("path=", path, "current_data=", current_data, ",all_data=", len(path_list), "")
            file = open(path, "r")
            file_list = file.readlines()
            temp_result_list = []
            position_list = []
            bottom = len(file_list) - 2
            for i in range(5, 2081):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
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
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
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
            temp = torch.cat((temp, position_tensor), 1)
            solute_data = temp[:173 * 12, :]
            solvent_data = temp[173 * 12:, :]
            solute_list.append(solute_data)  # 存入每个ACE文件的溶质图
            solvent_list.append(solvent_data)  # 存入每个ACE文件的溶剂图
            solute_position = position_tensor[:173 * 12, :]
            solvent_position = position_tensor[173 * 12:, :]
            dm = distance_matrix(solute_position, solvent_position)
            solute_solvent_adj = self.edge_ligand_pocket(dm, solute_data.shape[0])
            # edges= self.__cons_solute_solvent_graph_with_spatial_context__(solute_data[:,:-3], solute_position, solute_adj, solvent_data[:,:-3], solvent_position, solvent_adj)
            solute_solvent_adj_list.append(solute_solvent_adj)
        for i in range(len(lable_index)):
            lables_list.append(total_labels[lable_index[i]])  # 存入每个ACE文件的标签
        return solute_list, solvent_list, lables_list, solute_solvent_adj_list


    def __getitem__(self, index):
        solute_list_ACE = self.solute_list_ACE[index]
        solvent_list_ACE = self.solvent_list_ACE[index]
        lable_ACE = self.lables_list_ACE[index]
        solute_and_ACE_adj = self.solute_ACE_adj_list[index]

        solute_list_NMF = self.solute_list_NMF[index]
        solvent_list_NMF = self.solvent_list_NMF[index]
        lable_NMF = self.lables_list_NMF[index]
        solute_and_NMF_adj = self.solute_NMF_adj_list[index]

        solute_list_wat = self.solute_list_wat[index]
        solvent_list_wat = self.solvent_list_wat[index]
        lable_wat = self.lables_list_wat[index]
        solute_and_wat_adj = self.solute_wat_adj_list[index]

        return solute_list_ACE, solvent_list_ACE, lable_ACE, solute_and_ACE_adj, solute_list_NMF, solvent_list_NMF, lable_NMF, solute_and_NMF_adj, solute_list_wat, solvent_list_wat, lable_wat, solute_and_wat_adj

    def __len__(self):
        return len(self.solute_list_ACE)


def load_data(root_path_ACE, batch_size, phase, root_path_NMF, label_index,root_path_wat):
    data = trainset(root_path_ACE, phase, root_path_NMF,label_index, root_path_wat)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader

def load_test_data(batch_size, phase, label_index, root_path_meth):
    data = test_trainset(phase, label_index, root_path_meth)
    print("data=", data)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader

def val(net, val_loader, solute_adj, solvent_adj_ACE, solvent_adj_NMF, solvent_adj_wat):
    net.eval()
    SUMACE1 = 0
    SUMACE2 = 0
    SUMACE3 = 0
    SUMACE4 = 0
    SUMACE5 = 0

    SUMNMF1 = 0
    SUMNMF2 = 0
    SUMNMF3 = 0
    SUMNMF4 = 0
    SUMNMF5 = 0

    SUMwat1 = 0
    SUMwat2 = 0
    SUMwat3 = 0
    SUMwat4 = 0
    SUMwat5 = 0
    for i_batch, batch_data in tqdm(enumerate(val_loader)):
        solute_data_ACE, solvent_data_ACE, labels_ACE, solute_and_ACE_adj, solute_data_NMF, solvent_data_NMF, labels_NMF, solute_and_NMF_adj, solute_data_wat, solvent_data_wat, labels_wat, solute_and_wat_adj = batch_data
        labels_ACE = labels_ACE.to(DEVICE)
        labels_NMF = labels_NMF.to(DEVICE)
        labels_wat = labels_wat.to(DEVICE)
        labels = torch.cat((labels_ACE, labels_NMF), 0)
        labels = torch.cat((labels, labels_wat), 0)
        solute_data_ACE = solute_data_ACE.to(DEVICE)
        solute_data_NMF = solute_data_NMF.to(DEVICE)
        solute_data_wat = solute_data_wat.to(DEVICE)
        solvent_data_ACE = solvent_data_ACE.to(DEVICE)
        solvent_data_NMF = solvent_data_NMF.to(DEVICE)
        solvent_data_wat = solvent_data_wat.to(DEVICE)
        solute_adj = solute_adj.to(DEVICE)
        solvent_adj_ACE = solvent_adj_ACE.to(DEVICE)
        solvent_adj_NMF = solvent_adj_NMF.to(DEVICE)
        solvent_adj_wat = solvent_adj_wat.to(DEVICE)
        solute_and_ACE_adj = solute_and_ACE_adj.to(DEVICE)
        solute_and_NMF_adj = solute_and_NMF_adj.to(DEVICE)
        solute_and_wat_adj = solute_and_wat_adj.to(DEVICE)
        with torch.set_grad_enabled(False):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
            outputs = net(solute_data_ACE, solvent_data_ACE, solute_adj, solvent_adj_ACE, solute_data_NMF,
                          solvent_data_NMF, solvent_adj_NMF, solute_data_wat, solvent_data_wat, solvent_adj_wat, solute_and_ACE_adj, solute_and_NMF_adj, solute_and_wat_adj).squeeze(-1)
        deviation = abs((outputs - labels) / labels)
        print("outputs=", outputs)
        print("labels=", labels)
        print("deviation=", deviation)
        for i in range(deviation.shape[0]):
            if (i <= 3):
                if (deviation[i] < 0.01):
                    SUMACE1 = SUMACE1 + 1
                if (deviation[i] < 0.02):
                    SUMACE2 = SUMACE2 + 1
                if (deviation[i] < 0.03):
                    SUMACE3 = SUMACE3 + 1
                if (deviation[i] < 0.04):
                    SUMACE4 = SUMACE4 + 1
                if (deviation[i] < 0.05):
                    SUMACE5 = SUMACE5 + 1
            elif (i > 3 and i <= 7):
                if (deviation[i] < 0.01):
                    SUMNMF1 = SUMNMF1 + 1
                if (deviation[i] < 0.02):
                    SUMNMF2 = SUMNMF2 + 1
                if (deviation[i] < 0.03):
                    SUMNMF3 = SUMNMF3 + 1
                if (deviation[i] < 0.04):
                    SUMNMF4 = SUMNMF4 + 1
                if (deviation[i] < 0.05):
                    SUMNMF5 = SUMNMF5 + 1
            elif (i > 7):
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
    print("SUMACE1 = ", SUMACE1, "SUMACE2 = ", SUMACE2, "SUMACE3 = ", SUMACE3, "SUMACE4 = ", SUMACE4, "SUMACE5 = ",
          SUMACE5)
    print("SUMNMF1 = ", SUMNMF1, "SUMNMF2 = ", SUMNMF2, "SUMNMF3 = ", SUMNMF3, "SUMNMF4 = ", SUMNMF4, "SUMNMF5 = ",
          SUMNMF5)
    print("SUMwat1 = ", SUMwat1, "SUMwat2 = ", SUMwat2, "SUMwat3 = ", SUMwat3, "SUMwat4 = ", SUMwat4,
          "SUMwat5 = ", SUMwat5)
    print("tatol", (SUMACE5 * 100) / (len(val_loader) * 4), "% ACE in 5%", "tatol",
          (SUMNMF5 * 100) / (len(val_loader) * 4), "% NMF in 10%", "tatol",
          (SUMwat5 * 100) / (len(val_loader) * 4), "% Water in 10%")


def test(test_model, test_loader, solute_adj, solvent_adj_meth, PATH1, bestSUMmeth5, bestSUMmeth10, best_PATH5, best_PATH10):
    test_model.eval()
    SUMmeth1 = 0
    SUMmeth2 = 0
    SUMmeth3 = 0
    SUMmeth4 = 0
    SUMmeth5 = 0
    SUMmeth6 = 0
    SUMmeth7 = 0
    SUMmeth8 = 0
    SUMmeth9 = 0
    SUMmeth10 = 0

    for i_batch, batch_data in tqdm(enumerate(test_loader)):
        solute_data_meth, solvent_data_meth, labels_meth, solute_and_meth_adj = batch_data
        labels = labels_meth.to(DEVICE)
        solute_data_meth = solute_data_meth.to(DEVICE)
        solvent_data_meth = solvent_data_meth.to(DEVICE)
        solute_adj = solute_adj.to(DEVICE)
        solvent_adj_meth = solvent_adj_meth.to(DEVICE)
        solute_and_meth_adj = solute_and_meth_adj.to(DEVICE)
        with torch.set_grad_enabled(False):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
            outputs = test_model(solute_adj, solute_data_meth, solvent_data_meth, solvent_adj_meth, solute_and_meth_adj).squeeze(-1)
        deviation = abs((outputs - labels) / labels)
        print("outputs=", outputs)
        print("labels=", labels)
        print("deviation=", deviation)
        for i in range(deviation.shape[0]):
            if (i <= 3):
                if (deviation[i] < 0.01):
                    SUMmeth1 = SUMmeth1 + 1
                if (deviation[i] < 0.02):
                    SUMmeth2 = SUMmeth2 + 1
                if (deviation[i] < 0.03):
                    SUMmeth3 = SUMmeth3 + 1
                if (deviation[i] < 0.04):
                    SUMmeth4 = SUMmeth4 + 1
                if (deviation[i] < 0.05):
                    SUMmeth5 = SUMmeth5 + 1
                if (deviation[i] < 0.06):
                    SUMmeth6 = SUMmeth6 + 1
                if (deviation[i] < 0.07):
                    SUMmeth7 = SUMmeth7 + 1
                if (deviation[i] < 0.08):
                    SUMmeth8 = SUMmeth8 + 1
                if (deviation[i] < 0.09):
                    SUMmeth9 = SUMmeth9 + 1
                if (deviation[i] < 0.1):
                    SUMmeth10 = SUMmeth10 + 1
    if (SUMmeth10 > bestSUMmeth10):
        bestSUMmeth10 = SUMmeth10
        best_PATH10 = PATH1
    if (SUMmeth5 > bestSUMmeth5):
        bestSUMmeth5 = SUMmeth5
        best_PATH5 = PATH1
    print("SUMmeth1 = ", SUMmeth1, "SUMmeth2 = ", SUMmeth2, "SUMmeth3 = ", SUMmeth3, "SUMmeth4 = ", SUMmeth4,
          "SUMmeth5 = ", SUMmeth5)
    print("SUMmeth6 = ", SUMmeth6, "SUMmeth7 = ", SUMmeth7, "SUMmeth8 = ", SUMmeth8, "SUMmeth9 = ", SUMmeth9,
          "SUMmeth10 = ", SUMmeth10)
    print("tatol", (SUMmeth5*100)/(len(test_loader)*4), "% meth in 5%", "tatol", (SUMmeth10*100)/(len(test_loader)*4), "% meth in 10%")
    return best_PATH5, best_PATH10, bestSUMmeth5, bestSUMmeth10

# def get_solute_solvent_adj(edge_list, solvent):
#     if(solvent=="ACE"):
#         solute_solvent_adj = torch.zeros(11016,11016)
#     if (solvent == "meth"):
#         solute_solvent_adj = torch.zeros(18411, 18411)
#     if (solvent == "ACE"):
#         solute_solvent_adj = torch.zeros(14226, 14226)
#     if (solvent == "ACE"):
#         solute_solvent_adj = torch.zeros(16860, 16860)
#     for i,j in edge_list:
#         solute_solvent_adj[i][j] = 1
#     return  solute_solvent_adj

def train(train_loader, solute_adj, solvent_adj_ACE, solvent_adj_NMF, solvent_adj_wat, val_loader, solvent_adj_meth, test_loader):

    net = MyNewGCN(nfeat=9, nhid=64, nclass=128, dropout=args.dropout, DEVICE=DEVICE)
    # net = nn.DataParallel(net)
    net = net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.L1Loss()
    loss_list = []
    best_PATH5 = ""
    best_PATH10 = ""
    bestSUMmeth5 = 0
    bestSUMmeth10 = 0
    test_model = MyValModel(nfeat=9, nhid=64, nclass=128, dropout=args.dropout, DEVICE=DEVICE)
    test_model = test_model.to(DEVICE)
    for epoch in range(args.epoch):
        # tf_flag = 0
        net.train()
        total_loss = 0
        running_loss = 0
        for i_batch, batch_data in tqdm(enumerate(train_loader)):
            solute_data_ACE, solvent_data_ACE, labels_ACE, solute_and_ACE_adj, solute_data_NMF, solvent_data_NMF, labels_NMF, solute_and_NMF_adj, solute_data_wat, solvent_data_wat, labels_wat, solute_and_wat_adj = batch_data
            labels_ACE = labels_ACE.to(DEVICE)
            labels_NMF = labels_NMF.to(DEVICE)
            labels_wat = labels_wat.to(DEVICE)
            labels = torch.cat((labels_ACE, labels_NMF), 0)
            labels = torch.cat((labels, labels_wat), 0)
            solute_data_ACE = solute_data_ACE.to(DEVICE)
            solute_data_NMF = solute_data_NMF.to(DEVICE)
            solute_data_wat = solute_data_wat.to(DEVICE)
            solvent_data_ACE = solvent_data_ACE.to(DEVICE)
            solvent_data_NMF = solvent_data_NMF.to(DEVICE)
            solvent_data_wat = solvent_data_wat.to(DEVICE)
            solute_adj = solute_adj.to(DEVICE)
            solvent_adj_ACE = solvent_adj_ACE.to(DEVICE)
            solvent_adj_NMF = solvent_adj_NMF.to(DEVICE)
            solvent_adj_wat = solvent_adj_wat.to(DEVICE)
            solute_and_ACE_adj = solute_and_ACE_adj.to(DEVICE)
            solute_and_NMF_adj = solute_and_NMF_adj.to(DEVICE)
            solute_and_wat_adj = solute_and_wat_adj.to(DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                outputs = net(solute_data_ACE, solvent_data_ACE, solute_adj, solvent_adj_ACE, solute_data_NMF,
                              solvent_data_NMF, solvent_adj_NMF, solute_data_wat,
                              solvent_data_wat, solvent_adj_wat, solute_and_ACE_adj, solute_and_NMF_adj, solute_and_wat_adj).squeeze(-1)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item() * outputs.size(0)
            if (i_batch % 1000 == 9):
                print("outputs=", outputs)
                print("labels=", labels)
                print("loss=", loss)
                print('[epoch=%d, i_batch=%5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0
        writer.add_scalar('total loss', total_loss / len(train_loader), epoch)
        loss_list.append(total_loss / len(train_loader))
        PATH1 = "./check_point/" + str(epoch) + 'MyLastGCNTest8_net.pth'
        torch.save(net.state_dict(), PATH1)

        val(net, val_loader, solute_adj, solvent_adj_ACE, solvent_adj_NMF, solvent_adj_wat)

        test_model.load_state_dict(torch.load(PATH1))
        best_PATH5, best_PATH10, bestSUMmeth5, bestSUMmeth10 = test(test_model, test_loader, solute_adj, solvent_adj_meth,
                                                                   PATH1, bestSUMmeth5, bestSUMmeth10, best_PATH5,
                                                                   best_PATH10)
        print("best_PATH5=", best_PATH5, "best_PATH10=", best_PATH10)
    print("loss_list = ", loss_list)
    print('MyLastGCNTest8 Finished Training, lr=', args.lr, "batchsize=", args.batchsize)


# def test_new_data():
#
if __name__ == '__main__':
    data_list = []
    for i in range(20):
        data_list.append(i)
    train_lable_index, val_lable_index = train_test_split(data_list, train_size=0.6, random_state=42)
    dict_solute = get_solute_position(args.data_src_ACE)
    dict_solvent_NMF = get_solvent_position(args.data_src_NMF, "NMF")
    dict_solvent_ACE = get_solvent_position(args.data_src_ACE, "ACE")
    dict_solvent_wat = get_solvent_position(args.data_src_wat, "wat")
    dict_solvent_meth = get_solvent_position(args.data_src_meth, "meth")
    solute_adj = get_solute_adj()
    solvent_adj_NMF = get_solvent_adj("NMF", 12150)
    solvent_adj_ACE = get_solvent_adj("ACE", 8940)
    solvent_adj_wat = get_solvent_adj("wat", 14784)
    solvent_adj_meth = get_solvent_adj("meth", 16335)
    meth_atom_to_feature = np.load("./meth_atom_to_feature.npy", allow_pickle=True).item()
    wat_atom_to_feature = np.load("./wat_atom_to_feature.npy", allow_pickle=True).item()
    NMF_atom_to_feature = np.load("./NMF_atom_to_feature.npy", allow_pickle=True).item()
    ACE_atom_to_feature = np.load("./ACE_atom_to_feature.npy", allow_pickle=True).item()
    solute_addtional_feature = np.load("./solute_atom_to_feature.npy", allow_pickle=True).item()

    train_loader = load_data(args.data_src_ACE, args.batchsize, "train", args.data_src_NMF, train_lable_index, args.data_src_wat)#加载训练用的ACE+NMF+Water
    val_loader = load_data(args.data_src_ACE, args.batchsize, "val", args.data_src_NMF, val_lable_index, args.data_src_wat) #加载测试用的ACE+NMF+Water
    test_loader = load_test_data(args.batchsize, "test",data_list, args.data_src_meth) #加载meth

    train(train_loader, solute_adj, solvent_adj_ACE, solvent_adj_NMF, solvent_adj_wat, val_loader, solvent_adj_meth, test_loader)

    # test_new_data()





