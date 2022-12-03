import csv

import numpy as np
import pandas as pd

# meth_atom_to_feature = np.load("./meth_atom_to_feature.npy", allow_pickle=True).item()
# wat_atom_to_feature = np.load("./wat_atom_to_feature.npy", allow_pickle=True).item()
# NMF_atom_to_feature = np.load("./NMF_atom_to_feature.npy", allow_pickle=True).item()
# ACE_atom_to_feature = np.load("./ACE_atom_to_feature.npy", allow_pickle=True).item()
# solute_addtional_feature = np.load("./solute_atom_to_feature.npy", allow_pickle=True).item()

# pd.DataFrame(ACE_atom_to_feature).to_csv("ACE_atom_to_feature.csv")
# pd.DataFrame(meth_atom_to_feature).to_csv("meth_atom_to_feature.csv")
# pd.DataFrame(wat_atom_to_feature).to_csv("wat_atom_to_feature.csv")
# pd.DataFrame(NMF_atom_to_feature).to_csv("NMF_atom_to_feature.csv")
# pd.DataFrame(solute_addtional_feature).to_csv("solute_addtional_feature.csv")
data=np.load("./davis_train.npz",allow_pickle=True)
for item in data.files:
    print(data[item] )