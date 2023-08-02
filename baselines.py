import jax.numpy as jnp
from functools import partial
import jax
from utils import finance_utils


def polynomial(theta, X, g_X, theta_test, poly=3):
    """
    Polynomial Regression
    :param poly: int
    :param theta_test: N_test*D
    :param theta: T*D
    :param X: T*N*D
    :param g_X: T*N
    :return:
    """
    powers = jnp.arange(0, poly + 1)
    theta_poly = theta[:, :, None] ** powers
    theta_poly = theta_poly.reshape([theta.shape[0], -1])

    eps = 1.0
    D = (1 + poly) * theta.shape[1]
    theta = jnp.linalg.inv(theta_poly.T @ theta_poly + eps * jnp.eye(D)) @ theta_poly.T @ g_X.mean(1)

    theta_test_poly = theta_test[:, :, None] ** powers
    theta_test_poly = theta_test_poly.reshape([theta_test.shape[0], -1])
    phi = theta_test_poly @ theta
    std = 0
    pause = True
    return phi, std


# @partial(jax.jit, static_argnums=(0,))
def importance_sampling_sensitivity(log_pX_theta_fn, theta, X, g_X, theta_test):
    """
    :param log_pX_theta_fn:
    :param theta_test: T_test*D
    :param theta: T*D
    :param X: T*N*D
    :param g_X: T*N
    :return:
    """
    # log_pX_theta_test is (T, N, N_test)
    log_pX_theta_test = log_pX_theta_fn(X=X, theta=theta_test)
    # log_pX_theta_i is (T, N, T)
    log_pX_theta_i = log_pX_theta_fn(X=X, theta=theta)

    # log_pX_theta_test is (T, N, N_test)
    log_pX_theta_test = log_pX_theta_test.transpose(2, 0, 1)
    # log_pX_theta_i is (T, T, N)
    log_pX_theta_i = log_pX_theta_i.transpose(2, 0, 1)

    # weight is (N_test, T, N)
    weight = jnp.exp(log_pX_theta_test - jnp.diagonal(log_pX_theta_i, axis1=0, axis2=1).transpose(1, 0))
    # mu is (N_test, T)
    mu = (weight * g_X).mean(2)
    IS_mean = mu.mean(1)
    return IS_mean, 0


def importance_sampling_single_finance(tree, pX_theta_fn, theta_test):
    theta, Xi, g_Xi = tree
    Xi_standardized, Xi_mean, Xi_scale = finance_utils.standardize(Xi)
    pX_theta_standardized_fn = partial(pX_theta_fn, sigma=0.3, T=2, t=1, x_scale=Xi_scale, x_mean=Xi_mean)
    pX_theta_test = pX_theta_standardized_fn(Xi_standardized, theta_test)
    pX_theta_i = pX_theta_standardized_fn(Xi_standardized, theta)
    weight = pX_theta_test / pX_theta_i
    return (weight * g_Xi).mean() / weight.mean()


def importance_sampling_finance_(pX_theta_fn, theta, X, g_X, theta_test):
    theta_test = theta_test[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single_finance, pX_theta_fn=pX_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (theta, X, g_X)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_finance(pX_theta_fn, theta_test, theta, X, g_X):
    """
    Vectorized importance sampling for finance
    :param pX_theta_fn:
    :param theta_test: N_test*D
    :param theta: T*D
    :param X: T*N*D
    :param g_X: T*N
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_finance_, pX_theta_fn, theta, X, g_X)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_SIR(tree, log_py_theta_fn, theta_test):
    """
    :param log_py_x_fn:
    :param tree: consists of xi: scalar, Yi: (N, ), gYi: (N, )
    :param x_test:
    :return:
    """
    theta_i, Xi, gXi = tree
    log_py_theta_test = log_py_theta_fn(X=Xi, theta=theta_test)
    log_py_theta_i = log_py_theta_fn(X=Xi, theta=theta_i)
    weight = jnp.exp(log_py_theta_test - log_py_theta_i)
    mu = (weight * gXi).mean() / weight.mean()
    return mu


def importance_sampling_SIR_(log_py_theta_fn, X, Theta, gX, theta_test):
    importance_sampling_single_fn = partial(importance_sampling_single_SIR, log_py_theta_fn=log_py_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (Theta, X, gX)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_SIR(log_py_theta_fn, Theta_test, Theta, X, gX):
    """
    Vectorized importance sampling for SIR
    :param log_py_theta_fn:
    :param X_test: N_test*D
    :param X: T*D
    :param Y: T*N*D
    :param gY: T*N
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_SIR_, log_py_theta_fn, Theta, X, gX)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(Theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_decision(tree, py_x_fn, x_prime):
    x, Yi, gYi = tree
    Yi_standardized, Yi_mean, Yi_scale = decision_utils.standardize(Yi)
    py_x_standardized_fn = partial(py_x_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
    py_x_prime = py_x_standardized_fn(Yi_standardized, x_prime)
    py_x_i = py_x_standardized_fn(Yi_standardized, x)
    weight = py_x_prime / py_x_i
    return (weight * gYi).mean() / weight.mean()


def importance_sampling_decision_(py_x_fn, X, Y, gY, x_prime):
    x_prime = x_prime[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single_decision, py_x_fn=py_x_fn, x_prime=x_prime)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (X, Y, gY)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_decision(py_x_fn, X_prime, X, Y, gY):
    """
    Vectorized importance sampling
    :param py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_decision_, py_x_fn, X, Y, gY)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(X_prime)
    return IS_mean, 0 * IS_mean
