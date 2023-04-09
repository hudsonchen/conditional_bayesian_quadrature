import numpy as np
from utils import finance_utils
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt


def polynomial(X, Y, gY, X_prime, poly=3):
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    X_prime_standardized = (X_prime - X_mean) / X_std
    X_poly = jnp.ones_like(X_standardized)
    for i in range(1, poly + 1):
        X_poly = jnp.concatenate((X_poly, X_standardized ** i), axis=1)
    eps = 1e-6
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(poly + 1)) @ X_poly.T @ gY.mean(1)

    X_prime_poly = jnp.ones_like(X_prime_standardized)
    for i in range(1, poly + 1):
        X_prime_poly = jnp.concatenate((X_prime_poly, X_prime_standardized ** i), axis=1)
    phi = X_prime_poly @ theta

    std = jnp.zeros_like(phi)
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
            Yi_standardized, Yi_mean, Yi_scale = finance_utils.standardize(Yi)
            gYi = gY[i, :][:, None]

            py_x_standardized_fn = partial(py_x_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
            py_x_prime = py_x_standardized_fn(Yi_standardized, x_prime)
            py_x_i = py_x_standardized_fn(Yi_standardized, x)
            weight = py_x_prime / py_x_i
            dummy_list.append((weight * gYi).mean() / weight.mean())
        IS_list.append(np.array(dummy_list).mean())
        pause = True
    return jnp.array(IS_list), jnp.array(IS_list) * 0
