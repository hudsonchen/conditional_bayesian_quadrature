from utils import SIR_utils
from functools import partial
import jax.numpy as jnp
import jax


def polynomial(X, Y, gY, X_prime, poly=2):
    X_standardized, X_mean, X_std = SIR_utils.standardize(X)
    X_prime_standardized = (X_prime - X_mean) / X_std
    X_poly = jnp.ones_like(X_standardized)
    for i in range(1, poly + 1):
        X_poly = jnp.concatenate((X_poly, X_standardized ** i), axis=1)
    eps = 1e-3
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(poly + 1)) @ X_poly.T @ gY.mean(1)

    X_prime_poly = jnp.ones_like(X_prime_standardized)
    for i in range(1, poly + 1):
        X_prime_poly = jnp.concatenate((X_prime_poly, X_prime_standardized ** i), axis=1)
    phi = X_prime_poly @ theta
    std = jnp.zeros_like(phi)

    # Debugging code
    # plt.figure()
    # plt.plot(X_prime.squeeze(), phi.squeeze(), color='blue', label='predict')
    # plt.scatter(X.squeeze(), gY.mean(1).squeeze())
    # plt.legend()
    # plt.show()
    # pause = True
    return phi, std


def importance_sampling_(log_py_x_fn, X, Y, gY, x_prime):
    Nx, Ny = Y.shape
    x_prime = x_prime[:, None]
    dummy = jnp.zeros(Nx)
    for i in range(Nx):
        x = X[i]
        Yi = Y[i, :][:, None]
        gYi = gY[i, :][:, None]

        log_py_x_i = log_py_x_fn(beta=Yi, beta_0=x)
        log_py_x_prime = log_py_x_fn(beta=Yi, beta_0=x_prime)
        weight = jnp.exp(log_py_x_prime - log_py_x_i)
        dummy = dummy.at[i].set((weight * gYi).sum() / weight.sum())
    return dummy.mean()


def importance_sampling(log_py_x_fn, X_prime, X, Y, gY):
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


def importance_sampling_old(log_py_x_fn, X_prime, X, Y, gY):
    Nx, Ny = Y.shape
    IS_list = []
    for j in range(len(X_prime)):
        x_prime = X_prime[j]
        dummy_list = []
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :]
            gYi = gY[i, :]
            log_py_x_i = log_py_x_fn(beta=Yi, beta_0=x)
            log_py_x_prime = log_py_x_fn(beta=Yi, beta_0=x_prime)
            weight = jnp.exp(log_py_x_prime - log_py_x_i)
            dummy_list.append((weight * gYi).sum() / weight.sum())
        IS_list.append(jnp.array(dummy_list).mean())
    return jnp.array(IS_list), jnp.array(IS_list) * 0