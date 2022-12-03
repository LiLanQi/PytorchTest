import numpy as np
import torch

smile = torch.tensor(np.load("./smile.npy", allow_pickle=True))
print("smile=", smile)