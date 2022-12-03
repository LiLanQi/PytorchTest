import numpy as np
import rdkit.Chem.AllChem
import torch
from dgl._deprecate.graph import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

from testmodel import DataEmbedding

def one_of_k_encoding(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    for index,data in enumerate(allowable_set):
        if(x == data):
            return [index]
    return [len(allowable_set)]

def one_of_k_encoding_stereo(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    for index,data in enumerate(allowable_set):
        if(x == data):
            return [index+1]
    return [0]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    for index,data in enumerate(allowable_set):
        if(x == data):
            return [data]
    return [len(allowable_set)]


def get_len_matrix(len_list):
    len_list = np.array(len_list)

    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
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
    possible_atoms = ['H', 'C', 'O', 'N', 'S', 'F']

    atom_features = [atom.GetAtomicNum()]
    # print("atom.GetSymbol()=", atom.GetSymbol())
    # print("atom_features=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    # print("atom.GetImplicitValence()=", atom.GetImplicitValence())
    # print("atom_features=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    # print("atom.GetNumRadicalElectrons()=", atom.GetNumRadicalElectrons())
    # print("atom_features=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    # print("atom.GetDegree()=", atom.GetDegree())
    # print("atom_features=", atom_features)
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, -1])
    # print("atom.GetFormalCharge()=", atom.GetFormalCharge())
    # print("atom_features=", atom_features)
    atom_features += one_of_k_encoding_stereo(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    # print("atom.GetHybridization()=", atom.GetHybridization())
    # print("atom_features=", atom_features)
    # try:
    #     # print("stereo=", stereo)
    #     # atom_features += one_of_k_encoding_stereo(stereo, ['R', 'S'])
    #     # print("atom_features=", atom_features)
    # except Exception as e:
    #
    #     atom_features += [False, False
    #                       ] + [atom.HasProp('_ChiralityPossible')]

    return torch.tensor(np.array(atom_features))


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)

def get_graph_from_smile(molecule, atom_index, before_H_index):
    features = rdDesc.GetFeatureInvariants(molecule)
    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
    atom_to_feature = {}
    for i, atom_i in enumerate(molecule.GetAtoms()):
        print(atom_i.GetIdx(), end='\t')
        print(atom_i.GetAtomicNum(), end='\t')
        print(atom_i.GetSymbol(), end='\t')
        print(atom_i.GetDegree(), end='\t')
        print(atom_i.GetFormalCharge(), end='\t')
        print(atom_i.GetHybridization())

        if((i <= before_H_index) and (before_H_index != 0)):
            cur_atoms = atom_i.GetSymbol() +  str(atom_index[i])
        elif((before_H_index == 0) and (i == 0)):
            cur_atoms = atom_i.GetSymbol()
        else:
            cur_atoms = atom_i.GetSymbol() + str(i-before_H_index)
        # print(atom_i.GetSymbol(),"=",cur_atoms)
        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        atom_to_feature[cur_atoms] = atom_i_features
    return atom_to_feature

def get_additional_feature():
    solute = "FS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)cc1)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1"
    # solute = "O=S(OC(C=C1)=CC=C1C(C2=CC=C(OS(=O)(OC3=CC=C(C(C4=CC=C(OS(F)(=O)=O)C=C4)C5=CC=C(OS(=O)(F)=O)C=C5)C=C3)=O)C=C2)C6=CC=C(OS(OC7=CC=C(C(C8=CC=C(OS(F)(=O)=O)C=C8)C9=CC=C(OS(F)(=O)=O)C=C9)C=C7)(=O)=O)C=C6)(OC%10=CC=C(C(C%11=CC=C(OS(F)(=O)=O)C=C%11)C%12=CC=C(OS(F)(=O)=O)C=C%12)C=C%10)=O"
    mol = Chem.MolFromSmiles(solute)
    mol = Chem.AddHs(mol)
    solute_atom_index = [1, 1, 3, 4, 1, 3, 4, 5, 6, 19, 10, 11, 12, 7, 25, 8, 26, 27, 22, 64, 69, 68, 67, 76, 63, 62, 61, 60,
                  23, 7, 20, 21, 17, 55, 56, 51, 52, 57, 44, 43, 42, 41, 16, 6, 18, 19, 6, 40, 39, 48, 47, 46, 45, 13,
                  5, 14, 15, 5, 50, 49, 53, 54, 59, 58, 71, 70, 75, 74, 24, 9, 29, 30, 28, 26, 27, 28, 29, 38, 33, 32,
                  37, 36, 8, 4, 11, 12, 4, 35, 34, 25, 20, 21, 22, 7, 3, 9, 10, 3, 23, 24, 30, 31, 73, 72, 66, 65, 8, 9,
                  14, 13, 18, 17, 2, 2, 5, 6, 2, 16, 15, 1, 2]
    #
    print('\t'.join(['id', 'num', 'symbol', 'degree', 'charge', 'hybrid']))
    solute_atom_to_feature = get_graph_from_smile(mol, solute_atom_index, 120)

    # print("ACE=")
    ACE_atom_index = [1,2,1]
    ACE_solvent = "CC#N"
    mol = Chem.MolFromSmiles(ACE_solvent)
    mol = Chem.AddHs(mol)
    ACE_atom_to_feature = get_graph_from_smile(mol, ACE_atom_index, 2)
    #
    # print("NMF=")
    NMF_atom_index = [1, 1, 1, 2]
    NMF_solvent = "O=CNC"
    mol = Chem.MolFromSmiles(NMF_solvent)
    mol = Chem.AddHs(mol)
    NMF_atom_to_feature = get_graph_from_smile(mol, NMF_atom_index, 3)
    #
    wat_atom_index = []
    wat_solvent = "O"
    mol = Chem.MolFromSmiles(wat_solvent)
    mol = Chem.AddHs(mol)
    wat_atom_to_feature = get_graph_from_smile(mol, wat_atom_index, 0)

    meth_atom_index = [7,2,1,6,5,4,3]
    meth_solvent = "Cc1ccccc1"
    mol = Chem.MolFromSmiles(meth_solvent)
    mol = Chem.AddHs(mol)
    # print("mol=", Chem.MolToSmiles(mol))
    meth_atom_to_feature = get_graph_from_smile(mol, meth_atom_index, 6)

    DMF_atom_index = [1, 1, 1, 2, 3]
    DMF_solvent = "O=CN(C)C"
    mol = Chem.MolFromSmiles(DMF_solvent)
    mol = Chem.AddHs(mol)
    DMF_atom_to_feature = get_graph_from_smile(mol, DMF_atom_index, 4)

    return solute_atom_to_feature,ACE_atom_to_feature,NMF_atom_to_feature, wat_atom_to_feature, meth_atom_to_feature, DMF_atom_to_feature
    # wat_solvent = Chem.MolToSmiles(mol)


    # ACE_graph = get_graph_from_smile(ACE_solvent)
    # NMF_graph = get_graph_from_smile(NMF_solvent)
    # wat_graph = get_graph_from_smile(wat_solvent)
    #
    # return solute_graph, ACE_graph, NMF_graph, wat_graph

    # print("solute_graph.ndata['x'].float()=",solute_graph.ndata['x'].float().shape)
    # print("solute_graph.edata['w'].float()=",solute_graph.edata['w'].float().shape)
    # print("ACE_solvent_graph.ndata['x'].float()=",ACE_graph.ndata['x'].float().shape)
    # print("ACE_solvent_graph.edata['w'].float()=",ACE_graph.edata['w'].float().shape)
    # print("NMF_solvent_graph.ndata['x'].float()=",NMF_graph.ndata['x'].float().shape)
    # print("NMF_solvent_graph.edata['w'].float()=",NMF_graph.edata['w'].float().shape)
    # print("wat_solvent_graph.ndata['x'].float()=",wat_graph.ndata['x'].float().shape)
    # print("wat_solvent_graph.edata['w'].float()=",wat_graph.edata['w'].float().shape)
    #
    # model = DataEmbedding()
    # solute_enviroment_data, ACE_solvent_enviroment_data, NMF_solvent_enviroment_data, wat_solvent_enviroment_data = model(solute_graph.ndata['x'].float(), solute_graph.edata['w'].float(), ACE_graph.ndata['x'].float(), ACE_graph.edata['w'].float(),
    #       NMF_graph.ndata['x'].float(), NMF_graph.edata['w'].float(), wat_graph.ndata['x'].float())
    # PATH1 = './DataEmbedding.pth'
    # torch.save(model.state_dict(), PATH1)
    # return solute_enviroment_data, ACE_solvent_enviroment_data, NMF_solvent_enviroment_data, wat_solvent_enviroment_data

if __name__ == '__main__':
    solute_atom_to_feature,ACE_atom_to_feature,NMF_atom_to_feature, wat_atom_to_feature, meth_atom_to_feature,DMF_atom_to_feature = get_additional_feature()
    # print("ACE_atom_to_feature=", ACE_atom_to_feature)
    # print("ACE_atom_to_feature=", ACE_atom_to_feature)
    np.save('./solute_atom_to_feature.npy', solute_atom_to_feature)
    np.save('./ACE_atom_to_feature.npy', ACE_atom_to_feature)
    np.save('./NMF_atom_to_feature.npy', NMF_atom_to_feature)
    np.save('./wat_atom_to_feature.npy', wat_atom_to_feature)
    np.save('./meth_atom_to_feature.npy', meth_atom_to_feature)
    np.save('./DMF_atom_to_feature.npy', DMF_atom_to_feature)
    # get_additional_feature()
