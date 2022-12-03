import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbedding(nn.Module):
    def __init__(self):
        super(DataEmbedding, self).__init__()
        #solute
        self.fc1 = nn.Linear(42*121, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(2640, 512)
        self.fc4 = nn.Linear(512, 16)

        #ACE
        self.fc5 = nn.Linear(42 * 3, 42)
        self.fc6 = nn.Linear(42, 16)
        self.fc7 = nn.Linear(40, 20)
        self.fc8 = nn.Linear(20, 16)

        # NMF
        self.fc9 = nn.Linear(42 * 4, 42)
        self.fc10 = nn.Linear(42, 16)
        self.fc11 = nn.Linear(6*10, 20)
        self.fc12 = nn.Linear(20, 16)

        # water
        self.fc13 = nn.Linear(42 * 1, 20)
        self.fc14 = nn.Linear(20, 16)




    def forward(self, solute_ndata, solute_edata, ACE_solvent_ndata, ACE_solvent_edata, NMF_solvent_ndata, NMF_solvent_edata, wat_solvent_ndata):
        #solute
        solute_ndata = solute_ndata.reshape(1, -1)
        solute_ndata = F.relu(self.fc1(solute_ndata))
        solute_ndata = self.fc2(solute_ndata)

        solute_edata = solute_edata.reshape(1, -1)
        solute_edata = F.relu(self.fc3(solute_edata))
        solute_edata = self.fc4(solute_edata)

        #ACE
        ACE_solvent_ndata = ACE_solvent_ndata.reshape(1, -1)
        ACE_solvent_ndata = F.relu(self.fc5(ACE_solvent_ndata))
        ACE_solvent_ndata = self.fc6(ACE_solvent_ndata)

        ACE_solvent_edata = ACE_solvent_edata.reshape(1, -1)
        ACE_solvent_edata = F.relu(self.fc7(ACE_solvent_edata))
        ACE_solvent_edata = self.fc8(ACE_solvent_edata)

        # NMF
        NMF_solvent_ndata = NMF_solvent_ndata.reshape(1, -1)
        NMF_solvent_ndata = F.relu(self.fc9(NMF_solvent_ndata))
        NMF_solvent_ndata = self.fc10(NMF_solvent_ndata)

        NMF_solvent_edata = NMF_solvent_edata.reshape(1, -1)
        NMF_solvent_edata = F.relu(self.fc11(NMF_solvent_edata))
        NMF_solvent_edata = self.fc12(NMF_solvent_edata)

        # water
        wat_solvent_ndata = wat_solvent_ndata.reshape(1, -1)
        wat_solvent_ndata = F.relu(self.fc13(wat_solvent_ndata))
        wat_solvent_ndata = self.fc14(wat_solvent_ndata)
        wat_solvent_edata = torch.zeros(1, 16)

        solute_enviroment_data = torch.cat((solute_ndata, solute_edata), 1)
        ACE_solvent_enviroment_data = torch.cat((ACE_solvent_ndata, ACE_solvent_edata), 1)
        NMF_solvent_enviroment_data = torch.cat((NMF_solvent_ndata, NMF_solvent_edata), 1)
        wat_solvent_enviroment_data = torch.cat((wat_solvent_ndata, wat_solvent_edata), 1)

        # print("solute_ndata.shape=", solute_ndata.shape, "solute_edata.shape=", solute_edata.shape, "solute_enviroment_data=",solute_enviroment_data)
        # print("ACE_solvent_ndata.shape=", ACE_solvent_ndata.shape, "ACE_solvent_edata.shape=", ACE_solvent_edata.shape, "ACE_solvent_enviroment_data=",ACE_solvent_enviroment_data)
        # print("NMF_solvent_ndata.shape=", NMF_solvent_ndata.shape, "NMF_solvent_edata.shape=", NMF_solvent_edata.shape,"NMF_solvent_enviroment_data=", NMF_solvent_enviroment_data)
        # print("wat_solvent_ndata.shape=", wat_solvent_ndata.shape, "wat_solvent_edata.shape=", wat_solvent_edata.shape,"wat_solvent_enviroment_data=", wat_solvent_enviroment_data)

        return solute_enviroment_data, ACE_solvent_enviroment_data, NMF_solvent_enviroment_data, wat_solvent_enviroment_data
