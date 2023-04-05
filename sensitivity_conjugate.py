import jax
import jax.numpy as jnp
import os
import pwd
import jax.scipy
import jax.scipy.stats
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import MCMC
import argparse
from sensitivity_baselines import *
from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
from kernels import *
import optax
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
    return post_mean, post_cov


def g(y, x_star, noise):
    """
    :param y: w is a N_MCMC * D * 1 array
    """
    return y.sum(1).squeeze(axis=-1)


def Monte_Carlo(gy):
    return gy.mean(0)


# @jax.jit
def Bayesian_Monte_Carlo(rng_key, y, gy, d_log_py, kernel_y):
    """
    :param rng_key:
    :param y: N * D * 1
    :param gy: N
    :param d_log_py: N * D * 1
    :param kernel_y: kernel function
    :return:
    """
    y = y[:, :, 0]
    N, D = y.shape[0], y.shape[1]
    d_log_py = d_log_py[:, :, 0]
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    median_d = jnp.median(distance(y, y))
    gy_var = gy.var()
    c_init = c = 1.0 * gy_var
    log_l_init = log_l = jnp.log(median_d / jnp.sqrt(D))
    A_init = A = 1.0 * gy_var
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l, c, A = jnp.exp(log_l), c, A
        n = y.shape[0]
        K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps))
        return nll

    @jax.jit
    def step(log_l, c, A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
        log_l, c, A = optax.apply_updates((log_l, c, A), updates)
        return log_l, c, A, opt_state, nllk_value

    # # Debug code
    # log_l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(2000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # # Debug code
        # if jnp.isnan(nllk_value):
        #     p = 1
    #     log_l_debug_list.append(log_l)
    #     c_debug_list.append(c)
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(log_l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    l, c, A = jnp.exp(log_l), c, A
    final_K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c
    final_K_inv = jnp.linalg.inv(final_K + eps * jnp.eye(n))
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)

    if jnp.isnan(BMC_std):
        BMC_std = 0.3
    pause = True
    return BMC_mean, BMC_std


@jax.jit
def GP(psi_y_x_mean, psi_y_x_std, X, x_prime):
    """
    :param psi_y_x_mean: n_alpha*1
    :param psi_y_x_std: n_alpha*1
    :param X: n_alpha*3
    :param x_prime: 1*3
    :return:
    """
    Nx = psi_y_x_mean.shape[0]
    Mu_standardized, Mu_mean, Mu_std = finance_utils.standardize(psi_y_x_mean)
    Sigma_standardized = psi_y_x_std / Mu_std
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std
    noise = 0.01
    lx = 0.5

    K_train_train = my_RBF(X_standardized, X_standardized, lx) + jnp.diag(
        Sigma_standardized) + noise * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(x_prime_standardized, X_standardized, lx)
    K_test_test = my_RBF(x_prime_standardized, x_prime_standardized, lx) + noise
    mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
    var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std #+ jnp.mean(psi_y_x_std)
    return mu_y_x_prime_original, std_y_x_prime_original


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_covariance = 5.0
    noise = 0.1
    X, Y = generate_data(rng_key, D, 20, noise)
    # X = jnp.load(f'./data/sensitivity/data_x.npy')
    # Y = jnp.load(f'./data/sensitivity/data_y.npy')

    N_alpha_list = [5, 6]
    # N_alpha_list = [3, 5, 10, 20, 30]
    # N_beta_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N_beta_list = [100]

    cbq_mean_dict = {}
    cbq_std_dict = {}
    poly_mean_dict = {}
    poly_std_dict = {}
    IS_mean_dict = {}
    IS_std_dict = {}

    # This is the test point
    alpha_test = jax.random.uniform(rng_key, shape=(D, 1), minval=-1.0, maxval=1.0)
    cov_test = jnp.array([[prior_covariance] * D]).T + alpha_test

    post_mean, post_var = posterior_full(X, Y, cov_test, noise)
    g_test_true = post_mean.sum()

    for n_alpha in N_alpha_list:
        rng_key, _ = jax.random.split(rng_key)
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)
        # This is X, size n_alpha * D
        cov_all = jnp.array([[prior_covariance] * D]) + alpha_all
        cbq_mean_array = jnp.array([])
        cbq_std_array = jnp.array([])
        poly_mean_array = jnp.array([])
        poly_std_array = jnp.array([])
        IS_mean_array = jnp.array([])
        IS_std_array = jnp.array([])

        post_mean, post_var = posterior_full(X, Y, cov_test, noise)

        for n_beta in tqdm(N_beta_list):
            psi_mean_array = jnp.array([])
            psi_std_array = jnp.array([])
            logging = sensitivity_utils.init_logging()

            # This is Y and g(Y)
            states = jax.random.multivariate_normal(rng_key, mean=post_mean, cov=post_var, shape=(n_beta, D))

            for i in range(n_alpha):
                rng_key, _ = jax.random.split(rng_key)
                ind = jax.random.permutation(rng_key, len(states_all[f'{i}']))[:n_beta]
                states_i = states_all[f'{i}'][ind, :, :]
                g_states_i = g(states_i)
                g_states_i_standardized, g_states_i_scale = sensitivity_utils.scale(g_states_i)
                states = states.at[i, :, :, :].set(states_i)
                g_states = g_states.at[i, :].set(g_states_i)
                d_log_pstates = grad_log_prob(states_i)

                psi_mean, psi_std = Bayesian_Monte_Carlo(rng_key, states_i, g_states_i_standardized, d_log_pstates,
                                                         stein_Gaussian)
                psi_mean_array = jnp.append(psi_mean_array, psi_mean * g_states_i_scale)
                psi_std_array = jnp.append(psi_std_array, psi_std * g_states_i_scale)

                cov = cov_all[i, :][:, None]
                post_mean, post_var = posterior_full(X, Y, cov, noise)
                true_value = post_mean.sum()
                BMC_value = psi_mean * g_states_i_scale
                MC_value = g_states_i.mean()
                # # Debug
                print('True value', true_value)
                print(f'MC with {n_beta} number of Y', MC_value)
                print(f'BMC with {n_beta} number of Y', BMC_value)
                print(f"=================")
                pause = True
                logging = sensitivity_utils.update_log(args, n_alpha, n_beta, logging,
                                                       true_value, MC_value, BMC_value)


            BMC_mean, BMC_std = GP(psi_mean_array, psi_std_array, cov_all, cov_test.T)
            cbq_mean_array = jnp.append(cbq_mean_array, BMC_mean)
            cbq_std_array = jnp.append(cbq_std_array, BMC_std)

            mu_y_x_prime_poly, std_y_x_prime_poly = polynomial(cov_all, states, g_states, cov_test.T)
            poly_mean_array = jnp.append(poly_mean_array, mu_y_x_prime_poly)
            poly_std_array = jnp.append(poly_std_array, std_y_x_prime_poly)

            py_x_fn = partial(posterior, X=X, Y=Y, noise=noise)
            mu_y_x_prime_IS, std_y_x_prime_IS = importance_sampling(py_x_fn, cov_all, states, g_states, cov_test)
            IS_mean_array = jnp.append(IS_mean_array, mu_y_x_prime_IS)
            IS_std_array = jnp.append(IS_std_array, std_y_x_prime_IS)

        cbq_mean_dict[f"{n_alpha}"] = cbq_mean_array
        cbq_std_dict[f"{n_alpha}"] = cbq_std_array
        poly_mean_dict[f"{n_alpha}"] = poly_mean_array
        poly_std_dict[f"{n_alpha}"] = poly_std_array
        IS_mean_dict[f"{n_alpha}"] = IS_mean_array
        IS_std_dict[f"{n_alpha}"] = IS_std_array

    MC_list = []
    for Ny in N_beta_list:
        rng_key, _ = jax.random.split(rng_key)
        MC_list.append(g(states_test[:Ny, :]).mean())
    sensitivity_utils.save_final_results(args, MC_list, cbq_mean_dict, cbq_std_dict, poly_mean_dict,
                                         IS_mean_dict, g_test_true, N_alpha_list, N_beta_list)
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
