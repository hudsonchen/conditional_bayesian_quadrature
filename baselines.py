import jax.numpy as jnp
from functools import partial
import jax
from utils import finance_utils


def polynomial(theta, X, f_X, theta_test, baseline_use_variance, poly=3):
    """
    Polynomial Regression
    :param poly: int
    :param theta_test: T_Test*D
    :param theta: T*D
    :param X: T*N*D
    :param f_X: T*N
    :return:
    """
    powers = jnp.arange(0, poly + 1)
    theta_poly = theta[:, :, None] ** powers
    theta_poly = theta_poly.reshape([theta.shape[0], -1])

    if not baseline_use_variance:
        eps = 1e-6
        D = (1 + poly) * theta.shape[1]
        theta = jnp.linalg.inv(theta_poly.T @ theta_poly + eps * jnp.eye(D)) @ theta_poly.T @ f_X.mean(1)
    else:
        eps = 1e-6
        MC_std = f_X.std(1)
        D = (1 + poly) * theta.shape[1]
        theta = jnp.linalg.inv(theta_poly.T @ jnp.diag(MC_std) @ theta_poly + eps * jnp.eye(D)) @ (theta_poly.T @ jnp.diag(MC_std) @ f_X.mean(1))
    
    theta_test_poly = theta_test[:, :, None] ** powers
    theta_test_poly = theta_test_poly.reshape([theta_test.shape[0], -1])
    I_LSMC_mean = theta_test_poly @ theta
    I_LSMC_std = 0
    pause = True
    return I_LSMC_mean, I_LSMC_std


def importance_sampling(log_px_theta_fn, Theta_test, Theta, X, f_X):
    """
    Importance Sampling
    :param log_px_theta_fn: function to evaluate log p(x | theta)
    :param theta_test: T_Test*D
    :param theta: T*D
    :param X: T*N*D
    :param f_X: T*N
    :return:
    """
    T, N, D = X.shape
    IS_list = []
    for j in range(len(Theta_test)):
        theta_test = Theta_test[j]
        dummy_list = []
        for i in range(T):
            theta = Theta[i]
            Xi = X[i, :]
            f_X_i = f_X[i, :]
            log_py_theta_i = log_px_theta_fn(X=Xi, theta=theta)
            log_py_theta_test = log_px_theta_fn(X=Xi, theta=theta_test)
            weight = jnp.exp(log_py_theta_test - log_py_theta_i)
            dummy_list.append((weight * f_X_i).sum())
        IS_list.append(jnp.array(dummy_list).mean())
    return jnp.array(IS_list), jnp.array(IS_list) * 0



# @jax.jit
def kernel_mean_shrinkage(rng_key, I_mean, I_std, X, X_test, eps, kernel_fn):
    """
    :param kernel_fn: Matern or RBF
    :param eps:
    :param I_mean: (T, )
    :param I_std: (T, )
    :param X: (T, D)
    :param X_test: (T_test, D)
    :return: mu_x_theta: (T_test, ), std_x_theta: (T_test, )
    """
    T, D = X.shape[0], X.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0]) * D

    if I_std is None:
        sigma_array = jnp.array([1.0, 0.1, 0.01, 0.001])
        A_array = jnp.array([10.0, 100.0, 300.0, 1000.0])
        nll_array = jnp.zeros([len(l_array), len(A_array), len(sigma_array)])
    else:
        sigma_array = jnp.array([0.0])
        A_array = 0 * l_array
        nll_array = jnp.zeros([len(l_array), 1])

    if I_std is None:
        for i, l in enumerate(l_array):
            for j, A in enumerate(A_array):
                for k, sigma in enumerate(sigma_array):
                    K = A * kernel_fn(X, X, l) + jnp.eye(T) * sigma
                    K_inv = jnp.linalg.inv(K)
                    nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(
                        jnp.linalg.det(K) + 1e-6)) / T
                    nll_array = nll_array.at[i, j].set(nll)
        min_index_flat = jnp.argmin(nll_array)
        i1, i2, i3 = jnp.unravel_index(min_index_flat, nll_array.shape)
        l = l_array[i1]
        A = A_array[i2]
        sigma = sigma_array[i3]
    else:
        for i, l in enumerate(l_array):
            K_no_scale = kernel_fn(X, X, l)
            A = I_mean.T @ K_no_scale @ I_mean / T
            A_array = A_array.at[i].set(A)
            K = A * kernel_fn(X, X, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
            K_inv = jnp.linalg.inv(K)
            nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(jnp.linalg.det(K) + 1e-6)) / T
            nll_array = nll_array.at[i].set(nll)

        l = l_array[jnp.argmin(nll_array)]
        A = A_array[jnp.argmin(nll_array)]

    if I_std is None:
        K_train_train = A * kernel_fn(X, X, l) + jnp.eye(T) * sigma
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * kernel_fn(X_test, X, l)
        K_test_test = A * kernel_fn(X_test, X_test, l) + jnp.eye(X_test.shape[0]) * sigma
    else:
        K_train_train = A * kernel_fn(X, X, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * kernel_fn(X_test, X, l)
        K_test_test = A * kernel_fn(X_test, X_test, l) + eps * jnp.eye(X_test.shape[0])
    mu_X_theta_test = K_test_train @ K_train_train_inv @ I_mean
    var_X_theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_X_theta_test = jnp.abs(var_X_theta_test)
    std_x_theta = jnp.sqrt(var_X_theta_test)
    pause = True
    return mu_X_theta_test, std_x_theta


# @partial(jax.jit, static_argnums=(0,))
def importance_sampling_sensitivity(log_pX_theta_fn, Theta_test, Theta, X, f_X):
    """
    :param log_pX_theta_fn:
    :param theta_test: T_test*D
    :param theta: T*D
    :param X: T*N*D
    :param f_X: T*N
    :return:
    """
    # log_pX_theta_test is (T, N, T_Test)
    log_pX_theta_test = log_pX_theta_fn(X=X, theta=Theta_test)
    # log_pX_theta_i is (T, N, T)
    log_pX_theta_i = log_pX_theta_fn(X=X, theta=Theta)

    # log_pX_theta_test is (T, N, T_Test)
    log_pX_theta_test = log_pX_theta_test.transpose(2, 0, 1)
    # log_pX_theta_i is (T, T, N)
    log_pX_theta_i = log_pX_theta_i.transpose(2, 0, 1)

    # weight is (T_Test, T, N)
    weight = jnp.exp(log_pX_theta_test - jnp.diagonal(log_pX_theta_i, axis1=0, axis2=1).transpose(1, 0))
    # mu is (T_Test, T)
    mu = (weight * f_X).mean(2)
    IS_mean = mu.mean(1)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_finance(tree, pX_theta_fn, theta_test):
    theta, Xi, f_Xi = tree
    Xi_standardized, Xi_mean, Xi_scale = finance_utils.standardize(Xi)
    pX_theta_standardized_fn = partial(pX_theta_fn, sigma=0.3, T_finance=2, t_finance=1, x_scale=Xi_scale, x_mean=Xi_mean)
    pX_theta_test = pX_theta_standardized_fn(Xi_standardized, theta_test)
    pX_theta_i = pX_theta_standardized_fn(Xi_standardized, theta)
    weight = pX_theta_test / pX_theta_i
    return (weight * f_Xi).mean() / weight.mean()


def importance_sampling_finance_(pX_theta_fn, theta, X, f_X, theta_test):
    theta_test = theta_test[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single_finance, pX_theta_fn=pX_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (theta, X, f_X)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_finance(pX_theta_fn, theta_test, theta, X, f_X):
    """
    Vectorized importance sampling for finance
    :param pX_theta_fn:
    :param theta_test: T_Test*D
    :param theta: T*D
    :param X: T*N*D
    :param f_X: T*N
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_finance_, pX_theta_fn, theta, X, f_X)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_SIR(tree, log_py_theta_fn, theta_test):
    """
    :param log_px_theta_fn:
    :param tree: consists of xi: scalar, Yi: (N, ), gYi: (N, )
    :param x_test:
    :return:
    """
    theta_i, Xi, f_Xi = tree
    log_py_theta_test = log_py_theta_fn(X=Xi, theta=theta_test)
    log_py_theta_i = log_py_theta_fn(X=Xi, theta=theta_i)
    weight = jnp.exp(log_py_theta_test - log_py_theta_i)
    mu = (weight * f_Xi).mean() / weight.mean()
    return mu


def importance_sampling_SIR_(log_py_theta_fn, X, Theta, f_X, theta_test):
    importance_sampling_single_fn = partial(importance_sampling_single_SIR, log_py_theta_fn=log_py_theta_fn, theta_test=theta_test)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (Theta, X, f_X)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_SIR(log_py_theta_fn, Theta_test, Theta, X, f_X):
    """
    Vectorized importance sampling for SIR
    :param log_py_theta_fn:
    :param X_test: T_Test*D
    :param X: T*D
    :param Y: T*N*D
    :param gY: T*N
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_SIR_, log_py_theta_fn, Theta, X, f_X)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(Theta_test)
    return IS_mean, 0 * IS_mean


def importance_sampling_single_decision(tree, px_theta_fn, x_prime):
    x, Yi, gYi = tree
    Yi_standardized, Yi_mean, Yi_scale = decision_utils.standardize(Yi)
    px_theta_standardized_fn = partial(px_theta_fn, sigma=0.3, T=2, t=1, y_scale=Yi_scale, y_mean=Yi_mean)
    px_theta_prime = px_theta_standardized_fn(Yi_standardized, x_prime)
    px_theta_i = px_theta_standardized_fn(Yi_standardized, x)
    weight = px_theta_prime / px_theta_i
    return (weight * gYi).mean() / weight.mean()


def importance_sampling_decision_(px_theta_fn, X, Y, gY, x_prime):
    x_prime = x_prime[:, None]
    importance_sampling_single_fn = partial(importance_sampling_single_decision, px_theta_fn=px_theta_fn, x_prime=x_prime)
    importance_sampling_single_vmap = jax.vmap(importance_sampling_single_fn, in_axes=((0, 0, 0),))
    tree = (X, Y, gY)
    dummy = importance_sampling_single_vmap(tree)
    return dummy.mean()


def importance_sampling_decision(px_theta_fn, X_prime, X, Y, gY):
    """
    Vectorized importance sampling
    :param px_theta_fn:
    :param X_prime: T_Test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    importance_sampling_fn = partial(importance_sampling_decision_, px_theta_fn, X, Y, gY)
    importance_sampling_vmap = jax.vmap(importance_sampling_fn)
    IS_mean = importance_sampling_vmap(X_prime)
    return IS_mean, 0 * IS_mean
