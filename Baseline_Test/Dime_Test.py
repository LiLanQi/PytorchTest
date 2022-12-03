import argparse
import os.path as osp

import torch
from tensorboardX import SummaryWriter

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNet


parser = argparse.ArgumentParser()
parser.add_argument('--use_dimenet_plus_plus', action='store_true')
parser.add_argument('--base_lr', type=float, default=1E-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
args = parser.parse_args()
writer = SummaryWriter('MyDime')



path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)

# DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
# 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
dataset.data.y = dataset.data.y[:, idx]

device = torch.device('cuda:1')

i = 0
for target in range(12):
    # Skip target \delta\epsilon, since it can be computed via
    # \epsilon_{LUMO} - \epsilon_{HOMO}:
    if target == 4:
        continue
    with torch.no_grad():
        model= DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                            num_bilinear=8, num_spherical=7, num_radial=6,
                            cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                            num_after_skip=2, num_output_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=8)
    criterion = torch.nn.L1Loss()

    maes = []
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()
        # print("data=", data)
        # print("data.z.shape=", data.z.shape)
        # print("data.z=", data.z)
        # print("data.pos.shape=", data.pos.shape)
        # print("data.pos=", data.pos)
        # print("data.batch.shape=", data.batch.shape)
        # print("data.batch=", data.batch)
        with torch.set_grad_enabled(True):
            pred = model(data.z, data.pos, data.batch)
            print("pred=", pred)
            print("data.y[:, target]=", data.y[:, target])
        mae = (pred.view(-1) - data.y[:, target]).abs()
        # print("mae=", mae)
        loss = criterion(pred.view(-1), data.y[:, target])
        loss.backward()
        print("loss=", loss.item())
        writer.add_scalar('total loss', loss , i)
        i = i + 1
        optimizer.step()
        maes.append(mae)

    mae = torch.cat(maes, dim=0)

    # Report meV instead of eV:
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')