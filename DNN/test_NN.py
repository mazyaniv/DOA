import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from matplotlib import pyplot as pltfrom
from functions_NN import *
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

if __name__ == "__main__":
    SNR = 5
    snap = 400
    teta_max = 60
    teta_min = 0
    N = 50000
    test_size = 200
    numepochs = 70

    M = 10  # Sensors
    Q_vec = [0, int(M / 2), M]
    D = 2  # Sources
    thresh_real = 0
    thresh_im = 0
    shift = 0  # If -C to C, 0 class is -C so shift=C

