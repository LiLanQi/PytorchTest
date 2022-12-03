import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import seaborn as sns
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from gensim.models import word2vec
import torch
#通过加载预训练的mol2vec，得到溶质与五种溶剂的embedding，维度为300
# solute = "O=S(OC(C=C1)=CC=C1C(C2=CC=C(OS(=O)(OC3=CC=C(C(C4=CC=C(OS(F)(=O)=O)C=C4)C5=CC=C(OS(=O)(F)=O)C=C5)C=C3)=O)C=C2)C6=CC=C(OS(OC7=CC=C(C(C8=CC=C(OS(F)(=O)=O)C=C8)C9=CC=C(OS(F)(=O)=O)C=C9)C=C7)(=O)=O)C=C6)(OC%10=CC=C(C(C%11=CC=C(OS(F)(=O)=O)C=C%11)C%12=CC=C(OS(F)(=O)=O)C=C%12)C=C%10)=O"
# ACE_solvent = "CC#N"
# NMF_solvent = "O=CNC"
# wat_solvent = "O"
# meth_solvent = "Cc1ccccc1"
# DMF_solvent = "O=CN(C)C"

aa_smis = ['O=S(OC(C=C1)=CC=C1C(C2=CC=C(OS(=O)(OC3=CC=C(C(C4=CC=C(OS(F)(=O)=O)C=C4)C5=CC=C(OS(=O)(F)=O)C=C5)C=C3)=O)C=C2)C6=CC=C(OS(OC7=CC=C(C(C8=CC=C(OS(F)(=O)=O)C=C8)C9=CC=C(OS(F)(=O)=O)C=C9)C=C7)(=O)=O)C=C6)(OC%10=CC=C(C(C%11=CC=C(OS(F)(=O)=O)C=C%11)C%12=CC=C(OS(F)(=O)=O)C=C%12)C=C%10)=O',
           'CC#N', 'O=CNC','O','Cc1ccccc1','O=CN(C)C']
aa_codes = ['solute', 'ACE', 'NMF', 'Water', 'meth', 'DMF']

# aa_smis = ['CC(N)C(=O)O', 'N=C(N)NCCCC(N)C(=O)O', 'NC(=O)CC(N)C(=O)O', 'NC(CC(=O)O)C(=O)O',
#           'NC(CS)C(=O)O', 'NC(CCC(=O)O)C(=O)O', 'NC(=O)CCC(N)C(=O)O', 'NCC(=O)O',
#           'NC(Cc1cnc[nH]1)C(=O)O', 'CCC(C)C(N)C(=O)O', 'CC(C)CC(N)C(=O)O', 'NCCCCC(N)C(=O)O',
#           'CSCCC(N)C(=O)O', 'NC(Cc1ccccc1)C(=O)O', 'O=C(O)C1CCCN1', 'NC(CO)C(=O)O',
#           'CC(O)C(N)C(=O)O', 'NC(Cc1c[nH]c2ccccc12)C(=O)O', 'NC(Cc1ccc(O)cc1)C(=O)O',
#           'CC(C)C(N)C(=O)O']
# aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
#             'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

aas = [Chem.MolFromSmiles(x) for x in aa_smis]
Draw.MolsToGridImage(aas, molsPerRow=5, useSVG=False, legends=aa_codes)

sentence_solute = mol2alt_sentence(aas[0], 0)
sentence_ACE = mol2alt_sentence(aas[1], 1)
sentence_NMF = mol2alt_sentence(aas[2], 1)
sentence_Water = mol2alt_sentence(aas[3], 1)
sentence_meth = mol2alt_sentence(aas[4], 1)
sentence_DMF = mol2alt_sentence(aas[5], 1)
model = word2vec.Word2Vec.load('./model_300dim.pkl')
print("sentence=", sentence_solute)
print("len(sentence)=", len(sentence_solute))

def get_feature(sentence):
    temp = 0
    for index, single_sentence in enumerate(sentence):
        if (index == 0):
            temp = torch.from_numpy(model.wv.word_vec(single_sentence))
        else:
            temp = torch.cat((temp, torch.from_numpy(model.wv.word_vec(single_sentence))), 0)
    temp = temp.reshape(-1, 300)
    print("temp.shape=", temp.shape)
    return temp
np.save('./solute_feature.npy', get_feature(sentence_solute)) #（121,300）
np.save('./ACE_feature.npy', get_feature(sentence_ACE)) #（6,300）
np.save('./NMF_feature.npy', get_feature(sentence_NMF)) #（8,300）
np.save('./Water_feature.npy', get_feature(sentence_Water)) #（1,300）
np.save('./meth_feature.npy', get_feature(sentence_meth)) #（14,300）
np.save('./DMF_feature.npy', get_feature(sentence_DMF)) #（10,300）

