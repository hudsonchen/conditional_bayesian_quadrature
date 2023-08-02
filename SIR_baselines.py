from utils import SIR_utils
from functools import partial
import jax.numpy as jnp
import jax


def importance_sampling_single(tree, log_py_theta_fn, theta_test):
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


def importance_sampling_(log_py_theta_fn, X, Theta, gX, theta_test):
    importance_sampling_single_fn = partial(importance_sampling_single, log_py_theta_fn=log_py_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (Theta, X, gX)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling(log_py_theta_fn, Theta_test, Theta, X, gX):
    """
    Vectorized importance sampling
    :param log_py_theta_fn:
    :param X_test: N_test*D
    :param X: T*D
    :param Y: T*N*D
    :param gY: T*N
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_, log_py_theta_fn, Theta, X, gX)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(Theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_old(log_py_theta_fn, X_test, Theta, X, gX):
    T, N = X.shape
    IS_list = []
    for j in range(len(X_test)):
        theta_test = X_test[j]
        dummy_list = []
        for i in range(T):
            theta = Theta[i]
            Xi = X[i, :]
            gXi = gX[i, :]
            log_py_theta_i = log_py_theta_fn(X=Xi, theta=theta)
            log_py_theta_test = log_py_theta_fn(X=Xi, theta=theta_test)
            weight = jnp.exp(log_py_theta_test - log_py_theta_i)
            dummy_list.append((weight * gXi).sum() / weight.sum())
        IS_list.append(jnp.array(dummy_list).mean())
    return jnp.array(IS_list), jnp.array(IS_list) * 0