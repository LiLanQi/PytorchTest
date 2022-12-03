import numpy as np


solute_speed_list_ACE = np.load("./solute_speed_list_ACE_for_all_system.npy", allow_pickle=True).item()
solvent_speed_list_ACE = np.load("./solvent_speed_list_ACE_for_all_system.npy", allow_pickle=True).item()
solute_solvent_speed_list_ACE = np.load("./solute_solvent_speed_list_ACE_for_all_system.npy", allow_pickle=True).item()
print(solute_speed_list_ACE)
print(solvent_speed_list_ACE)
print(solute_solvent_speed_list_ACE)
print(solute_speed_list_ACE[1])