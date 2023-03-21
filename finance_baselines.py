import numpy as np
from utils import finance_utils
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt


def polynomial(X, Y, gY, X_prime, poly=3):
    poly_list = []
    for j in range(len(X_prime)):
        x_prime = X_prime[j][:, None]
        X_standardized, X_mean, X_std = finance_utils.standardize(X)
        x_prime_standardized = (x_prime - X_mean) / X_std
        X_poly = np.ones_like(X_standardized)
        for i in range(1, poly + 1):
            X_poly = np.concatenate((X_poly, X_standardized ** i), axis=1)
        eps = 1e-6
        theta = np.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(poly + 1)) @ X_poly.T @ gY.mean(1)

        x_prime_poly = np.ones_like(x_prime_standardized)
        for i in range(1, poly + 1):
            x_prime_poly = np.concatenate((x_prime_poly, x_prime_standardized ** i), axis=1)
        phi = (theta * x_prime_poly).sum()
        poly_list.append(phi)
    phi = np.array(poly_list)
    std = np.zeros_like(phi)
    # Debugging code
    # true_X = jnp.load('./data/finance_X.npy')
    # true_EgY_X = jnp.load('./data/finance_EgY_X.npy')
    #
    # x_debug = np.linspace(20, 100, 100)[:, None]
    # x_debug_standardized = (x_debug - X_mean) / X_std
    # x_debug_poly = np.ones_like(x_debug_standardized)
    # for i in range(1, poly + 1):
    #     x_debug_poly = np.concatenate((x_debug_poly, x_debug_standardized ** i), axis=1)
    # mu_y_x_debug = (theta * x_debug_poly).sum(1)
    #
    # plt.figure()
    # plt.plot(x_debug.squeeze(), mu_y_x_debug.squeeze(), color='blue', label='predict')
    # plt.plot(true_X, true_EgY_X, color='red', label='true')
    # plt.scatter(X.squeeze(), gY.mean(1).squeeze())
    # plt.legend()
    # plt.show()
    pause = True
    return phi, std


def importance_sampling(py_x_fn, X_prime, X, Y, gY):
    Nx, Ny = Y.shape
    IS_list = []
    for j in range(len(X_prime)):
        x_prime = X_prime[j][:, None]
        dummy_list = []
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            Yi_standardized, Yi_scale, Yi_mean = finance_utils.scale(Yi)
            gYi = gY[i, :][:, None]

            py_x_standardized_fn = partial(py_x_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
            py_x_prime = py_x_standardized_fn(Yi_standardized, x_prime)
            py_x_i = py_x_standardized_fn(Yi_standardized, x)
            weight = py_x_prime / py_x_i
            dummy_list.append((weight * gYi).mean())
        IS_list.append(np.array(dummy_list).mean())
    return np.array(IS_list), np.array(IS_list) * 0
