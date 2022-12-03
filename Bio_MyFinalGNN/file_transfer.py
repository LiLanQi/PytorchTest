from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
# mol = AllChem.MolFromPDBFile("C:/Users/Administrator/Desktop/分子数据/5p21.pdb")
# DMF_Smiles = Chem.MolToSmiles()
# print(DMF_Smiles)
from rdkit.Chem import Draw
import os

def save_picture(smile, cur_type):
    mol = Chem.MolFromSmiles(smile)
    Draw.MolToImage(mol, size=(500, 500), kekulize=True)
    Draw.ShowMol(mol, size=(500, 500), kekulize=False)
    path = os.path.join('C:/Users/Administrator/Desktop/论文图片/', cur_type + ".png")
    print(path)
    Draw.MolToFile(mol, path, size=(500, 500))

def save_picture_haveH(smile, cur_type):
    m3d = Chem.MolFromSmiles(smile)
    m3d = Chem.AddHs(m3d)
    AllChem.EmbedMolecule(m3d, randomSeed=1, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(m3d)
    Draw.MolToImage(m3d, size=(250,250))
    Draw.ShowMol(m3d, size=(500,500), kekulize=False)
    path = os.path.join('C:/Users/Administrator/Desktop/论文图片/', cur_type + "_haveH.png")
    Draw.MolToFile(m3d, path, size=(500, 500))
    # cids = AllChem.EmbedMultipleConfs(m3d, numConfs=10)
    # Draw.ShowMol(cids, size=(500, 500), kekulize=False)

# solute = "FS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)c1ccc(OS(=O)(=O)Oc1ccc(C(c1ccc(OS(=O)(=O)F)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1)cc1)cc1)cc1)c1ccc(OS(=O)(=O)F)cc1)cc1"
solute = "O=S(OC(C=C1)=CC=C1C(C2=CC=C(OS(=O)(OC3=CC=C(C(C4=CC=C(OS(F)(=O)=O)C=C4)C5=CC=C(OS(=O)(F)=O)C=C5)C=C3)=O)C=C2)C6=CC=C(OS(OC7=CC=C(C(C8=CC=C(OS(F)(=O)=O)C=C8)C9=CC=C(OS(F)(=O)=O)C=C9)C=C7)(=O)=O)C=C6)(OC%10=CC=C(C(C%11=CC=C(OS(F)(=O)=O)C=C%11)C%12=CC=C(OS(F)(=O)=O)C=C%12)C=C%10)=O"
ACE_solvent = "CC#N"
NMF_solvent = "O=CNC"
wat_solvent = "O"
meth_solvent = "Cc1ccccc1"
DMF_solvent = "O=CN(C)C"
# save_picture(solute, "solute")
# save_picture(ACE_solvent, "ACE")
# save_picture(NMF_solvent, "NMF")
# save_picture(wat_solvent, "water")
# save_picture(meth_solvent, "meth")
# save_picture(DMF_solvent, "DMF")
# save_picture_haveH(solute, "solute")

m = Chem.MolFromSmiles('C1CCC1OC')
m2=Chem.AddHs(m)
# run ETKDG 10 times
AllChem.EmbedMolecule(m2)
AllChem.MMFFOptimizeMolecule(m2)
Draw.ShowMol(m2, size=(500, 500), kekulize=False)




