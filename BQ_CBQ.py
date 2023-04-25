import jax
import jax.numpy as jnp
from kernels import *
import time
from jax.config import config
import matplotlib.pyplot as plt
from tqdm import tqdm

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


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


@jax.jit
def Bayesian_Monte_Carlo(y, gy, mu_y_x, sigma_y_x):
    """
    :param sigma_y_x: covariance
    :param mu_y_x: mean
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6

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


@jax.jit
def GP(rng_key, psi_y_x_mean, psi_y_x_std, X, X_prime, eps):
    """
    :param eps:
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    n_alpha, D = X.shape[0], X.shape[1]
    l = 3.0
    K_no_scale = my_Matern(X, X, l)
    A = psi_y_x_mean.T @ K_no_scale @ psi_y_x_mean / n_alpha

    K_train_train = A * my_Matern(X, X, l) + eps * jnp.eye(n_alpha) + jnp.diag(psi_y_x_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * my_Matern(X_prime, X, l)
    K_test_test = A * my_Matern(X_prime, X_prime, l) + eps * jnp.eye(X_prime.shape[0])

    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_y_x = jnp.abs(var_y_x)
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def standardize(X):
    mean = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)
    X = (X - mean) / std
    return X, mean, std


def Bayesian_Monte_Carlo_fn(tree):
    y, gy, mu_y_x, sigma_y_x = tree
    return Bayesian_Monte_Carlo(y, gy, mu_y_x, sigma_y_x)


def main():
    # Compare the performance of BQ and CBQ
    seed = int(time.time())
    rng_key = jax.random.PRNGKey(seed)
    D = 2

    prior_cov_base = 2.0
    noise = 1.0
    data_number = 5
    # X is (N, D-1), Y is (N, 1)
    X, Y = generate_data(rng_key, D, data_number, noise)

    g = g3
    g_ground_truth_fn = g3_ground_truth

    n_theta_array = jnp.arange(10, 110, 10)

    time_BQ_array = 0. * n_theta_array
    time_CBQ_array = 0. * n_theta_array
    mse_BQ_array = 0. * n_theta_array
    mse_CBQ_array = 0. * n_theta_array

    for j, n_theta in enumerate(tqdm(n_theta_array)):
        n_alpha = n_theta

        rng_key, _ = jax.random.split(rng_key)
        # This is X, size n_alpha * D
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)
        # This is Y, size n_alpha * n_theta * D
        samples_all = jnp.zeros([n_alpha, n_theta, D])
        # This is g(Y), size n_alpha * sample_size
        g_samples_all = jnp.zeros([n_alpha, n_theta])
        prior_cov = jnp.array([[prior_cov_base] * D]) + alpha_all
        mu_y_x_all, var_y_x_all = posterior_full(X, Y, prior_cov, noise)

        alpha_test = alpha_all[0, :][None, :]

        for i in range(n_alpha):
            rng_key, _ = jax.random.split(rng_key)
            samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x_all[i, :], cov=var_y_x_all[i, :, :],
                                                     shape=(n_theta,))
            samples_all = samples_all.at[i, :, :].set(samples)
            g_samples_all = g_samples_all.at[i, :].set(g(samples))

        samples_all_BQ = samples_all.reshape([n_alpha * n_theta, D])
        rng_key, _ = jax.random.split(rng_key)
        permutation = jax.random.permutation(rng_key, n_alpha * n_theta)
        samples_all_BQ = samples_all_BQ[permutation, :]
        g_samples_all_BQ = g_samples_all.reshape([n_alpha * n_theta])
        g_samples_all_BQ = g_samples_all_BQ[permutation]

        mu_y_x_test, var_y_x_test = mu_y_x_all[0, :], var_y_x_all[0, :, :]

        # ==================== Ground Truth ====================
        ground_truth = g_ground_truth_fn(mu_y_x_test, var_y_x_test)

        # ==================== BQ ====================
        _, _ = Bayesian_Monte_Carlo(samples_all_BQ, g_samples_all_BQ,
                                    mu_y_x_test, var_y_x_test)
        t0 = time.time()
        BQ_mean, BQ_std = Bayesian_Monte_Carlo(samples_all_BQ, g_samples_all_BQ,
                                               mu_y_x_test, var_y_x_test)
        BQ_time = time.time() - t0
        BQ_mse = (BQ_mean - ground_truth) ** 2
        mse_BQ_array = mse_BQ_array.at[j].set(BQ_mse)
        time_BQ_array = time_BQ_array.at[j].set(BQ_time)

        # ==================== CBQ ====================
        rng_key, _ = jax.random.split(rng_key)
        psi_mean_array = jnp.zeros(n_alpha)
        psi_std_array = jnp.zeros(n_alpha)

        for i in range(n_alpha):
            samples_i = samples_all[i, :, :]
            g_samples_i = g_samples_all[i, :]
            mu_y_x_i = mu_y_x_all[i, :]
            var_y_x_i = var_y_x_all[i, :, :]

            t0 = time.time()
            psi_mean, psi_std = Bayesian_Monte_Carlo(samples_i, g_samples_i, mu_y_x_i, var_y_x_i)
            t1 = time.time()

            psi_mean_array = psi_mean_array.at[i].set(psi_mean)
            psi_std_array = psi_std_array.at[i].set(psi_std if not jnp.isnan(psi_std) else 0.01)

        _, _ = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test,
                  eps=psi_std_array.mean())

        t2 = time.time()
        CBQ_mean, CBQ_std = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test,
                               eps=psi_std_array.mean())
        t3 = time.time()
        CBQ_time = (t1 - t0) * n_alpha + (t3 - t2)

        CBQ_mean = CBQ_mean[0]
        CBQ_mse = (CBQ_mean - ground_truth) ** 2
        mse_CBQ_array = mse_CBQ_array.at[j].set(CBQ_mse)
        time_CBQ_array = time_CBQ_array.at[j].set(CBQ_time)

        # ============= Debug code =============
        # print("BQ mean: ", BQ_mean)
        # print("CBQ mean: ", CBQ_mean)
        # print("Ground truth: ", ground_truth)
        #
        # print("BQ time: ", BQ_time)
        # print("CBQ time: ", CBQ_time)
        # pause = True
        # ============= Debug code =============

    # ==================== Plot ====================

    jnp.save("./ablations/mse_BQ_array.npy", mse_BQ_array)
    jnp.save("./ablations/mse_CBQ_array.npy", mse_CBQ_array)
    jnp.save("./ablations/time_BQ_array.npy", time_BQ_array)
    jnp.save("./ablations/time_CBQ_array.npy", time_CBQ_array)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(n_theta_array, mse_BQ_array, label="BQ")
    axs[0].plot(n_theta_array, mse_CBQ_array, label="CBQ")
    axs[0].set_xlabel("Sample Numbers")
    axs[0].set_ylabel("MSE")
    axs[0].legend()
    axs[0].set_yscale("log")

    axs[1].plot(n_theta_array, time_BQ_array, label="BQ")
    axs[1].plot(n_theta_array, time_CBQ_array, label="CBQ")
    axs[1].set_xlabel("Sample Numbers")
    axs[1].set_ylabel("Time")
    axs[1].legend()
    axs[1].set_yscale("log")
    plt.show()
    pause = True


if __name__ == "__main__":
    main()
