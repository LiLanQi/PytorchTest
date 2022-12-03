from collections import defaultdict
import os
import pickle
import sys

import numpy as np

from rdkit import Chem
import networkx as nx


def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print("atoms=", atoms)
    # print("mol.GetAtoms()=",mol.GetAtoms())
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    # print("atoms1=", atoms)
    atoms = [atom_dict[a] for a in atoms]
    # print("atoms2=",atoms)
    # print("np.array(atoms)=", np.array(atoms))
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # print("i=", i, "j=", j)
        # print("str(b.GetBondType())=", str(b.GetBondType()))
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    # print("i_jbond_dict=",i_jbond_dict)
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        print("nodes=", nodes)
        print("i_jedge_dict=", i_jedge_dict)
        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                # print("i=", i, "j_edge=", j_edge)
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                # print("neighbors=",neighbors)
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                # print("fingerprint=", fingerprint)
                fingerprints.append(fingerprint_dict[fingerprint])
                # print("fingerprints=", fingerprints)
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            print("----------------------------------------------------------------------------------------")
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    # print("i=", i, "j_edge=", j_edge)
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    # print("both_side=", both_side)
                    edge = edge_dict[(both_side, edge)]
                    # print("edge=", edge)
                    _i_jedge_dict[i].append((j, edge))
                    # print("_i_jedge_dict=", _i_jedge_dict)
            i_jedge_dict = _i_jedge_dict
    # print("fingerprints=",fingerprints)
    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    print("adjacency=", adjacency)
    return np.array(adjacency)

def create_edge_index(mol):
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return edge_index

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

if __name__ == "__main__":

    # DATASET, radius, ngram = 'human', 2, 2
    radius, ngram = 2, 2

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    # print("atom_dict=",atom_dict) atom_dict= defaultdict(<function <lambda> at 0x7fd342cff9d8>, {})
    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    solute = "FS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)cc1)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1"
    ACE_solvent = "CC#N"
    NMF_solvent = "O=CNC"
    wat_solvent = "O"
    DMF="O=CN(C)C"
    meth_solvent = "Cc1ccccc1"

    mol = Chem.AddHs(Chem.MolFromSmiles(DMF))  # Consider hydrogens.
    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)

    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
    compounds.append(fingerprints)

    # To generate adjacent matrix from GNN and GAT method
    adjacency = create_adjacency(mol)
#         adjacency = create_edge_index(mol)
    print("adjacency.shape=",adjacency.shape)
    adjacencies.append(adjacency)
    print("fingerprints=", fingerprints)
    np.save('./DMF_solvent_compounds.npy', fingerprints)
