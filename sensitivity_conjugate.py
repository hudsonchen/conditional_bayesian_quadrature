import jax
import jax.numpy as jnp
import os
import shutil
import pwd
import jax.scipy
import jax.scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sensitivity_baselines import *
from tqdm import tqdm
from kernels import *
from utils import finance_utils, sensitivity_utils
import time
from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
else:
    pass

eps = 1e-6

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


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


@jax.jit
def posterior_full(X, Y, prior_cov, noise):
    """
    :param prior_cov: D*1 array
    :param X: N*(D-1) array
    :param Y: N*1 array
    :param noise: float
    :return:
    """
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    prior_cov = jnp.diag(prior_cov.squeeze())
    prior_cov_inv = jnp.diag(1. / prior_cov.squeeze())
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * X_with_one.T @ X_with_one)
    post_mean = beta * post_cov @ X_with_one.T @ Y
    return post_mean.squeeze(), post_cov


@jax.jit
def posterior_llk(theta, prior_cov_base, X, Y, alpha, noise):
    """
    :param theta: (N, D)
    :param prior_cov_base: scalar
    :param X: data
    :param Y: data
    :param alpha: (D, 1)
    :param noise: scalar
    :return:
    """
    D = theta.shape[1]
    prior_cov = jnp.array([[prior_cov_base] * D]).T + alpha
    post_mean, post_cov = posterior_full(X, Y, prior_cov, noise)
    return jax.scipy.stats.multivariate_normal.pdf(theta, post_mean, post_cov)


def g(y):
    """
    :param y: y is a N * D array
    """
    return y.sum(1)


def Monte_Carlo(gy):
    return gy.mean(0)


# @jax.jit
def Bayesian_Monte_Carlo(rng_key, y, gy, mu_y_x, sigma_y_x):
    """
    :param rng_key:
    :param y: N * D
    :param gy: N
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    eps = 1e-6
    l = 1.0

    K = my_RBF(y, y, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = kme_RBF_Gaussian(mu_y_x, sigma_y_x, l, y)
    varphi = kme_double_RBF_Gaussian(mu_y_x, sigma_y_x, l)

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def GP(psi_y_x_mean, psi_y_x_std, X, X_prime):
    """
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    Nx = psi_y_x_mean.shape[0]
    Sigma = psi_y_x_std
    eps = 0.01
    lx = 5.0

    K_train_train = my_RBF(X, X, lx) + jnp.diag(Sigma) + eps * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(X_prime, X, lx)
    K_test_test = my_RBF(X_prime, X_prime, lx) + eps
    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_cov_base = 5.0
    noise = 1.0
    sample_size = 1000
    test_num = 100
    X, Y = generate_data(rng_key, D, 10, noise)

    N_alpha_list = [5, 10, 20]
    # N_alpha_list = [3, 5, 10, 20, 30]
    # N_theta_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N_theta_list = [50, 100, 200]

    # This is the test point
    alpha_test_line = jax.random.uniform(rng_key, shape=(test_num, D), minval=-2.0, maxval=2.0)
    cov_test_line = jnp.array([[prior_cov_base] * D]) + alpha_test_line
    post_mean_test_line, post_var_test_line = jnp.zeros([test_num, D]), jnp.zeros([test_num, D, D])
    for i in range(test_num):
        post_mean, post_var = posterior_full(X, Y, cov_test_line[i, :], noise)
        post_mean_test_line = post_mean_test_line.at[i, :].set(post_mean)
        post_var_test_line = post_var_test_line.at[i, :, :].set(post_var)
    ground_truth = post_mean_test_line.sum(1)
    jnp.save(f"{args.save_path}/test_line.npy", alpha_test_line)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    for n_alpha in N_alpha_list:
        rng_key, _ = jax.random.split(rng_key)
        # This is X, size n_alpha * D
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)

        # This is Y, size n_alpha * sample_size * D
        samples_all = jnp.zeros([n_alpha, sample_size, D])
        # This is g(Y), size n_alpha * sample_size
        g_samples_all = jnp.zeros([n_alpha, sample_size])
        mu_y_x_all = jnp.zeros([n_alpha, D])
        var_y_x_all = jnp.zeros([n_alpha, D, D])

        for i in range(n_alpha):
            rng_key, _ = jax.random.split(rng_key)
            prior_cov = jnp.array([[prior_cov_base] * D]).T + alpha_all[i, :]
            mu_y_x, var_y_x = posterior_full(X, Y, prior_cov, noise)
            samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x, cov=var_y_x, shape=(1000, ))
            samples_all = samples_all.at[i, :, :].set(samples)
            g_samples_all = g_samples_all.at[i, :].set(g(samples))
            mu_y_x_all = mu_y_x_all.at[i, :].set(mu_y_x)
            var_y_x_all = var_y_x_all.at[i, :, :].set(var_y_x)

        for n_theta in tqdm(N_theta_list):
            psi_mean_array = jnp.array([])
            psi_std_array = jnp.array([])
            mc_mean_array = jnp.array([])

            for i in range(n_alpha):
                rng_key, _ = jax.random.split(rng_key)
                samples_i = samples_all[i, :n_theta, :]
                g_samples_i = g_samples_all[i, :n_theta]
                g_samples_i_standardized, g_samples_i_scale = sensitivity_utils.scale(g_samples_i)
                mu_y_x_i = mu_y_x_all[i, :]
                var_y_x_i = var_y_x_all[i, :, :]

                psi_mean, psi_std = Bayesian_Monte_Carlo(rng_key, samples_i, g_samples_i_standardized, mu_y_x_i, var_y_x_i)
                psi_mean_array = jnp.append(psi_mean_array, psi_mean * g_samples_i_scale)
                psi_std_array = jnp.append(psi_std_array, psi_std * g_samples_i_scale)

                MC_value = g_samples_i.mean()
                mc_mean_array = jnp.append(mc_mean_array, MC_value)

                # # # Debug
                # true_value = mu_y_x_i.sum()
                # BMC_value = psi_mean * g_samples_i_scale
                # print("=============")
                # print('True value', true_value)
                # print(f'MC with {n_theta} number of Y', MC_value)
                # print(f'BMC with {n_theta} number of Y', BMC_value)
                # print(f"=============")

            BMC_mean, BMC_std = GP(psi_mean_array, psi_std_array, alpha_all, alpha_test_line)
            KMS_mean, KMS_std = GP(mc_mean_array, mc_mean_array * 0, alpha_all, alpha_test_line)
            LSMC_mean, LSMC_std = polynomial(alpha_all, samples_all[:, :n_theta, :],
                                             g_samples_all[:, :n_theta], alpha_test_line)
            py_x_fn = partial(posterior_llk, X=X, Y=Y, noise=noise, prior_cov_base=prior_cov_base)
            IS_mean, IS_std = importance_sampling(py_x_fn, alpha_all, samples_all[:, :n_theta, :],
                                                  g_samples_all[:, :n_theta], alpha_test_line)
            sensitivity_utils.save(args, n_alpha, n_theta, BMC_mean, BMC_std, KMS_mean, KMS_std, LSMC_mean,
                                   LSMC_std, IS_mean, IS_std)

            mse_BMC = jnp.mean((BMC_mean - ground_truth) ** 2)
            mse_KMS = jnp.mean((KMS_mean - ground_truth) ** 2)
            mse_LSMC = jnp.mean((LSMC_mean - ground_truth) ** 2)
            mse_IS = jnp.mean((IS_mean - ground_truth) ** 2)
            print(f"=============")
            print(f"MSE of BMC with {n_alpha} number of X and {n_theta} number of Y", mse_BMC)
            print(f"MSE of KMS with {n_alpha} number of X and {n_theta} number of Y", mse_KMS)
            print(f"MSE of LSMC with {n_alpha} number of X and {n_theta} number of Y", mse_LSMC)
            print(f"MSE of IS with {n_alpha} number of X and {n_theta} number of Y", mse_IS)
            print(f"=============")
            pause = True
    return


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')

    # Args settings
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    return args


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity_conjugate/'
    args.save_path += f"seed_{args.seed}__dim_{args.dim}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    create_dir(args)
    print(f'Device is {jax.devices()}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
