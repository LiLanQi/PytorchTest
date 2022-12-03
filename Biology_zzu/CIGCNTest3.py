#encoding=utf-8
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os

import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit import Chem
import numpy as np
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--result_src', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5)

DEVICE = torch.device('cuda:0')
args = parser.parse_args()
writer = SummaryWriter('GCNTest3')

dict_solute = {}
dict_solvent = {}
solute = "FS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)cc1)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1"
solute_smiles_str = "F1,S1,O3,O4,O1,C3,C4,C5,C6,C19,C10,C11,C12,C7,O25,S8,O26,O27,O22,C64,C69,C68,C67,C76,C63,C62,C61,C60,O23,S7,O20,O21,O17,C55,C56,C51,C52,C57,C44,C43,C42,C41,O16,S6,O18,O19,F6,C40,C39,C48,C47,C46,C45,O13,S5,O14,O15,F5,C50,C49,C53,C54,C59,C58,C71,C70,C75,C74,O24,S9,O29,O30,O28,C26,C27,C28,C29,C38,C33,C32,C37,C36,O8,S4,O11,O12,F4,C35,C34,C25,C20,C21,C22,O7,S3,O9,O10,F3,C23,C24,C30,C31,C73,C72,C66,C65,C8,C9,C14,C13,C18,C17,O2,S2,O5,O6,F2,C16,C15,C1,C2"
mol = Chem.MolFromSmiles(solute)
mol = Chem.AddHs(mol)
solute = Chem.MolToSmiles(mol)
solvent = "CC(=C)C"
mol = Chem.MolFromSmiles(solvent)
mol = Chem.AddHs(mol)
solvent = Chem.MolToSmiles(mol)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
        print("not=")

    return list(map(lambda s: x == s, allowable_set))


def get_len_matrix(len_list):
    print("len_list1=",len_list)
    len_list = np.array(len_list)

    max_nodes = np.sum(len_list)
    print("len_list=",len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    print("len_matrix=", len_matrix)
    return np.array(len_matrix)

def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Si']
    # print("atom.GetSymbol()=",atom.GetSymbol())
    # print("atom.GetImplicitValence()=", atom.GetImplicitValence())
    # print("atom.GetNumRadicalElectrons()=", atom.GetNumRadicalElectrons())
    # print("atom.GetDegree()=", atom.GetDegree())
    # print("atom.GetFormalCharge()=", atom.GetFormalCharge())
    # print("atom.GetHybridization()=", atom.GetHybridization())
    # print("Chem.rdchem.HybridizationType.SP=", Chem.rdchem.HybridizationType.SP)
    # print("Chem.rdchem.HybridizationType.SP2=", Chem.rdchem.HybridizationType.SP2)
    # print("Chem.rdchem.HybridizationType.SP3=", Chem.rdchem.HybridizationType.SP3)
    # print("Chem.rdchem.HybridizationType.SP3D=", Chem.rdchem.HybridizationType.SP3D)
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    # print("atom_features1=",atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    # print("atom_features2=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    # print("atom_features3=", atom_features)
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    # print("atom_features4=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    # print("atom_features5=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    # print("atom_features6=", atom_features)
    # print("features=",features)
    # print(list("{0:06b}".format(features)))将feature表示成2进制
    atom_features += [int(i) for i in list("{0:06b}".format(features))]
    # print("atom_features7=", atom_features)

    if not explicit_H:
        # print("111")
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    # print("atom_features8=", atom_features)
    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        # print("atom_features9=", atom_features)
        # print("hasprop=",atom.HasProp('_ChiralityPossible'))
        atom_features += [atom.HasProp('_ChiralityPossible')]
        # print("atom_features10=", atom_features)
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]
        # print("atom_features11=", atom_features)
    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    # print("bond_type=",bond_type)
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    # print("bond.GetStereo()=",bond.GetStereo())
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    # print("bond_feats=",bond_feats)
    return np.array(bond_feats)


def get_graph_from_smile(molecule_smile,position_data):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    G = DGLGraph()
    molecule = Chem.MolFromSmiles(molecule_smile)
    features = rdDesc.GetFeatureInvariants(molecule)
    print("features=",features)

    stereo = Chem.FindMolChiralCenters(molecule)
    print("stereo=",stereo)
    chiral_centers = [0] * molecule.GetNumAtoms()
    print("chiral_centers=",chiral_centers)
    for i in stereo:
        chiral_centers[i[0]] = i[1]
        print("i=",i)
    print("G=",G)
    G.add_nodes(molecule.GetNumAtoms())
    print("molecule.GetNumAtoms()=",molecule.GetNumAtoms())
    print("G=", G)
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)
        print("atom_i")
        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        print("atom.GetSymbol()=",atom_i.GetSymbol())
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            # print("bond_ij=",bond_ij)
            if bond_ij is not None:
                G.add_edge(i, j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)
    print("node_features=",node_features)
    print("edge_features=", edge_features)
    G.ndata['x'] = torch.from_numpy(np.array(node_features))
    G.edata['w'] = torch.from_numpy(np.array(edge_features))
    return G

def get_solute_position(rootdir):
    path_list = readFileData(rootdir)
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
    path_list = readFileData(rootdir)
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
    for i in range(0, 1000):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result


def readFileData(root_dir):
    path_list = []
    for i in range(0, 1000):
        temp_path = "ACE" + str(i) + ".pdb"
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
    solvent_adj = solvent_adj + torch.eye(6*1490, 6*1490)
    return solvent_adj

def append_str_data(C,S,F,O):
    datas = []
    index = 0
    atom_datas = str.split(solute_smiles_str,",")
    for atom in atom_datas:
        if("C" == atom[0]):
            datas[index] = C[int(atom[1:])]
        elif("O" == atom[0]):
            datas[index] = O[int(atom[1:])]
        elif ("S" == atom[0]):
            datas[index] = S[int(atom[1:])]
        elif ("F" == atom[0]):
            datas[index] = F[int(atom[1:])]
        index = index+1
    return datas

class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase):
        self.root = rootdir
        solute_list = []
        solvent_list = []
        path_list = readFileData(rootdir)  # 得到所有mdx.pdb文件的list




        for path in path_list:
            file = open(path, "r")
            file_list = file.readlines()
            temp_result_list = []
            C = torch.zeros(1,80)
            F = torch.zeros(1,10)
            S = torch.zeros(1,10)
            O = torch.zeros(1,40)
            print("O.shape=",O.shape)
            print("O=", O)
            for i in range(5, 173 + 5):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
                single_atom = temp_atom[0]
                temp_atom_index = int(temp_atom[1:])
                if ("H" == single_atom):
                    continue
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                position_data = torch.tensor((x,y,z))
                if("C" == single_atom):
                    C[0][temp_atom_index] = position_data
                elif("F" == single_atom):
                    F[0][temp_atom_index] = position_data
                elif("S" == single_atom):
                    S[0][temp_atom_index] = position_data
                elif("O" == single_atom):
                    print("temp_atom_index=",temp_atom_index)
                    O[0][temp_atom_index] = position_data
            print("O=", O)
            print("C=", C)
            print("F=", F)
            print("S=", S)
            solute_data = append_str_data(C,S,F,O)
            print("solute_data=",solute_data)
            print("type(solute_data)=", type(solute_data))
            break
            solute_graph = get_graph_from_smile(solute,solute_data)

            # solvent_graph = get_graph_from_smile(solvent,solvent_data)
            solute_list.append(solute_graph)  # 存入每一个mdx 文件的所有原子坐标，直到所有mdx 文件的所有原子坐标都存入进来
            # solvent_list.append(solvent_graph)
        self.solute_list = solute_list
        # self.solvent_list = solvent_list
        self.lables = readFileResult(args.result_src)  # 存入所有mdx 文件对应的output值

    def __getitem__(self, index):
        solute_graph = self.solute_list[index]
        # solvent_graph = self.solvent_list[index]
        lable = self.lables[index]
        return solute_graph,  lable

    def __len__(self):
        return len(self.solute_list)


def load_train(root_path, batch_size, result_src, phase):
    data = trainset(root_path, result_src, phase)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
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


if __name__ == '__main__':

    data_loader = load_train(args.data_src, args.batchsize, args.result_src, "train")



