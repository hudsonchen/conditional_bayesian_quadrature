import numpy as np
from utils import finance_utils
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt


def polynomial(X, Y, gY, x_prime, poly=4):
    """
    Polynomial Regression
    :param x_prime: 1*3
    :param X: Nx*3
    :param Y: Nx*Ny*3*1
    :param gY: Nx*Ny
    :return:
    """
    X_poly = np.ones_like(X)
    for i in range(1, poly + 1):
        X_poly = np.concatenate((X_poly, X ** i), axis=1)
    eps = 1.0
    D = (1 + poly) * X.shape[1]
    theta = np.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(D)) @ X_poly.T @ gY.mean(1)

    x_prime_poly = np.ones_like(x_prime)
    for i in range(1, poly + 1):
        x_prime_poly = np.concatenate((x_prime_poly, x_prime ** i), axis=1)
    phi = (theta * x_prime_poly).sum()
    std = 0
    pause = True
    return phi, std


def importance_sampling(py_x_fn, X, Y, gY, x_prime):
    """
    Self-normalized importance sampling
    :param py_x_fn:
    :param x_prime: 3*1
    :param X: Nx*3
    :param Y: Nx*Ny*3*1
    :param gY: Nx*Ny
    :return:
    """
    Nx, Ny = Y.shape[0], Y.shape[1]
    IS_list = []
    for i in range(Nx):
        x = X[i, :][:, None]
        Yi = Y[i, :, :, :]
        gYi = gY[i, :][:, None]
        py_x_prime = py_x_fn(beta=Yi, prior_cov=x_prime)
        py_x_i = py_x_fn(beta=Yi, prior_cov=x)
        weight = py_x_prime / py_x_i
        mu = (weight * gYi).mean() / (weight.mean())
        IS_list.append(mu)
    IS = jnp.array(IS_list).mean()
    return IS, 0
