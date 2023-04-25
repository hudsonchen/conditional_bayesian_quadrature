import jax
import jax.numpy as jnp
from kernels import *


def posterior_full(X, Y, prior_cov, noise):
    """
    :param prior_cov: (N3, D)
    :param X: (N, D-1)
    :param Y: (N, 1)
    :param noise: float
    :return: (N3, D), (N3, D, D)
    """
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    D = prior_cov.shape[-1]
    prior_cov_inv = 1. / prior_cov
    # (N3, D, D)
    prior_cov_inv = jnp.einsum('ij,jk->ijk', prior_cov_inv, jnp.eye(D))
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * X_with_one.T @ X_with_one)
    post_mean = beta * post_cov @ X_with_one.T @ Y
    return post_mean.squeeze(), post_cov

def generate_data(rng_key, D, N, noise):
    """
    :param rng_key:
    :param D: int
    :param N: int
    :param noise: std for Gaussian likelihood
    :return: X is N*(D-1), Y is N*1
    """
    rng_key, _ = jax.random.split(rng_key)
    X = jax.random.uniform(rng_key, shape=(N, D - 1), minval=-1.0, maxval=1.0)
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    rng_key, _ = jax.random.split(rng_key)
    beta_true = jax.random.normal(rng_key, shape=(D, 1))
    rng_key, _ = jax.random.split(rng_key)
    Y = X_with_one @ beta_true + jax.random.normal(rng_key, shape=(N, 1)) * noise
    return X, Y

def g3(y):
    return (y ** 2).sum(1)


def g3_ground_truth(mu, Sigma):
    return jnp.diag(Sigma).sum() + mu.T @ mu

def Bayesian_Monte_Carlo(rng_key, y, gy, mu_y_x, sigma_y_x):
    """
    :param mu_y_x:
    :param rng_key:
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

    l_array = jnp.array([0.1, 0.6, 1.0, 3.0]) * D
    nll_array = 0 * l_array
    A_list = []

    for i, l in enumerate(l_array):
        K_no_scale = my_RBF(y, y, l)
        A = gy.T @ K_no_scale @ gy / N
        A_list.append(A)
        K = A * K_no_scale
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / N
        nll_array = nll_array.at[i].set(nll)

    if D > 2:
        l = l_array[nll_array.argmin()]
        A = A_list[nll_array.argmin()]
    else:
        A = 1.
        l = 1.

    K = A * my_RBF(y, y, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_y_x, sigma_y_x, l, y)
    varphi = A * kme_double_RBF_Gaussian(mu_y_x, sigma_y_x, l)

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)
    pause = True
    return BMC_mean, BMC_std


def main():
    # Compare the performance of BQ and CBQ
    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    D = 2

    prior_cov_base = 2.0
    noise = 1.0
    sample_size = 5000
    test_num = 200
    data_number = 5
    # X is (N, D-1), Y is (N, 1)
    X, Y = generate_data(rng_key, D, data_number, noise)

    g = g3
    g_ground_truth_fn = g3_ground_truth

    n_alpha = 10
    n_theta = 10

    rng_key, _ = jax.random.split(rng_key)
    # This is X, size n_alpha * D
    alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)
    # This is Y, size n_alpha * n_theta * D
    samples_all = jnp.zeros([n_alpha, n_theta, D])
    # This is g(Y), size n_alpha * sample_size
    g_samples_all = jnp.zeros([n_alpha, n_theta])
    prior_cov = jnp.array([[prior_cov_base] * D]) + alpha_all
    mu_y_x_all, var_y_x_all = posterior_full(X, Y, prior_cov, noise)

    alpha_test = alpha_all[0, :]

    for i in range(n_alpha):
        rng_key, _ = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x_all[i, :], cov=var_y_x_all[i, :, :],
                                                 shape=(n_theta,))
        samples_all = samples_all.at[i, :, :].set(samples)
        g_samples_all = g_samples_all.at[i, :].set(g(samples))

    samples_all_BQ = samples_all.reshape([n_alpha * n_theta, D])
    g_samples_all_BQ = g_samples_all.reshape([n_alpha * n_theta])

    mu_y_x_test, var_y_x_test = mu_y_x_all[0, :], var_y_x_all[0, :, :]
    BQ_mean, BQ_std = Bayesian_Monte_Carlo(rng_key, y, gy, mu_y_x_test, sigma_y_x_test)
