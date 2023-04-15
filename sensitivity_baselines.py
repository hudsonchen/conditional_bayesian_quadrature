import jax.numpy as jnp
from functools import partial
import jax


def polynomial(X, Y, gY, X_prime, poly=3):
    """
    Polynomial Regression
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    X_poly = jnp.ones_like(X)
    for i in range(1, poly + 1):
        X_poly = jnp.concatenate((X_poly, X ** i), axis=1)
    eps = 1.0
    D = (1 + poly) * X.shape[1]
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(D)) @ X_poly.T @ gY.mean(1)

    X_prime_poly = jnp.ones_like(X_prime)
    for i in range(1, poly + 1):
        X_prime_poly = jnp.concatenate((X_prime_poly, X_prime ** i), axis=1)
    phi = X_prime_poly @ theta
    std = 0
    pause = True
    return phi, std


def importance_sampling_(log_py_x_fn, X, Y, gY, x_prime):
    Nx, Ny = Y.shape[0], Y.shape[1]
    temp_array = jnp.zeros([Nx])
    for i in range(Nx):
        xi = X[i, :]
        Yi = Y[i, :, :]
        gYi = gY[i, :]
        log_py_x_prime = log_py_x_fn(theta=Yi, alpha=x_prime)
        log_py_x_i = log_py_x_fn(theta=Yi, alpha=xi)
        weight = jnp.exp(log_py_x_prime - log_py_x_i)
        mu = (weight * gYi).mean() / (weight.mean() + 0.1)
        temp_array = temp_array.at[i].set(mu)
    return temp_array.mean()


def importance_sampling(log_py_x_fn, X, Y, gY, X_prime):
    """
    Vectorized importance sampling
    :param log_py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_, log_py_x_fn, X, Y, gY)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(X_prime)
    return IS_mean, 0 * IS_mean


def importance_sampling_old(log_py_x_fn, X, Y, gY, X_prime):
    """
    :param log_py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    Nx, Ny = Y.shape[0], Y.shape[1]
    N_test = X_prime.shape[0]
    IS_prime_list = []
    for j in range(N_test):
        IS_list = []
        x_prime = X_prime[j, :]
        for i in range(Nx):
            xi = X[i, :]
            Yi = Y[i, :, :]
            gYi = gY[i, :]
            log_py_x_prime = log_py_x_fn(theta=Yi, alpha=x_prime)
            log_py_x_i = log_py_x_fn(theta=Yi, alpha=xi)
            weight = jnp.exp(log_py_x_prime - log_py_x_i)
            mu = (weight * gYi).mean() / (weight.mean())
            IS_list.append(mu)
        IS_prime_list.append(jnp.array(IS_list).mean())
        pause = True
    return jnp.array(IS_prime_list), 0
