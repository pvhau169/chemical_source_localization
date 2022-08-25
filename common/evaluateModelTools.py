import torch
import numpy as np
def kernelSmooth(X ,y, sigma= 4):
    X, y = X[y>=0], y[y>=0]
    # y[y==-1] = 0
    grid = torch.linspace(torch.min(X), torch.max(X) * 1.01, int(torch.max(X) - torch.min(X)) * 2)

    K = torch.exp(- (X[:, None] - grid[None, :] )**2 / ( 2 *sigma **2))

    e_at_grid = y[None ,:] @ K/ torch.sum(K, 0)
    e_at_grid = e_at_grid.reshape(-1)

    return grid, e_at_grid