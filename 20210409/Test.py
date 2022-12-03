from tqdm import tqdm
import argparse
#import data_loader
from Dogset import load_train
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
from torchsummary import summary
import datetime
from tensorboardX import SummaryWriter
import os

if __name__=='__main__':
    print(torchvision.models.vgg19())
