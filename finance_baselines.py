import numpy as np
import torch
from torch.autograd import grad
from torch import optim
from utils import finance_utils


def polynommial(X, Y, gY, x_prime, sigma, poly=3):
    X = torch.tensor(np.asarray(X), dtype=torch.double)
    Y = torch.tensor(np.asarray(Y), dtype=torch.double)
    gY = torch.tensor(np.asarray(gY), dtype=torch.double)
    x_prime = torch.tensor(np.asarray(x_prime), dtype=torch.double)

    gY_standardized, gY_mean, gY_std = finance_utils.standardize(gY)
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std

    theta = torch.tensor([[1.0] * (poly + 1)], requires_grad=True, dtype=torch.double)
    optimizer = optim.Adam([theta], lr=0.01)
    v = torch.tensor([10.0])
    v_old = torch.tensor([-10.0])
    while torch.abs(v - v_old) > 0.01:
        optimizer.zero_grad()
        X_poly = np.ones_like(X_standardized)
        for i in range(1, poly + 1):
            X_poly = np.concatenate((X_poly, X_standardized ** i), axis=1)
        X_poly_tensor = torch.tensor(X_poly)
        gY_tensor = gY_standardized
        v_old = v
        v = torch.mean((X_poly_tensor @ theta.T - gY_tensor.mean(1)) ** 2)
        v.backward()
        optimizer.step()

    theta_array = theta.detach().numpy()
    x_prime_poly = np.ones_like(x_prime_standardized)
    for i in range(1, poly + 1):
        x_prime_poly = np.concatenate((x_prime_poly, x_prime_standardized ** i), axis=1)
    phi = (theta_array * x_prime_poly).sum()
    phi_original = phi * gY_std + gY_mean
    std = 0
    return phi_original.numpy(), std


def importance_sampling():
    pass