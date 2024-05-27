import torch.nn as nn
import torch.nn.functional as F

# ? Maybe construct the network in this class
class Prim3DModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

# ? or construct the network in this function
def _Prim3DModel():
    return nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
