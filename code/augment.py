import torch
import numpy as np

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))

    return lam*x + (1-lam)*x[idx], y, y[idx], lam


def cutmix(x, y):
    idx = torch.randperm(x.size(0))
    lam = 0.7

    x = x.clone()
    x[:, :, :, :40] = x[idx, :, :, :40]

    return x, y, y[idx], lam