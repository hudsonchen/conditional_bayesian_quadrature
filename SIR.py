import jax.numpy as jnp
import numpy as np
import jax
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from kernels import *
import optax
from tqdm import tqdm
from functools import partial
from utils import SIR_utils
from tensorflow_probability.substrates import jax as tfp
from jax.config import config
import SIR_baselines
import argparse
import os
import pwd
import shutil

tfd = tfp.distributions
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
else:
    pass


def Monte_Carlo(gy):
    return gy.mean(0)


def Bayesian_Monte_Carlo(rng_key, y, gy, d_log_py, kernel_y):
    """
    :param rng_key:
    :param y: N * D
    :param gy: N
    :param d_log_py: N * D
    :param kernel_y: kernel function
    :return:
    """
    n = y.shape[0]
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    c_init = c = 1.5
    l_init = l = 1.5
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((l_init, c_init, A_init))

    @jax.jit
    def nllk_func(l, c, A):
        l, c, A = l, c, A
        n = y.shape[0]
        K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
        return nll

    @jax.jit
    def step(l, c, A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A)
        updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
        l, c, A = optax.apply_updates((l, c, A), updates)
        return l, c, A, opt_state, nllk_value

    # # Debug code
    l_debug_list = []
    c_debug_list = []
    A_debug_list = []
    nll_debug_list = []
    for _ in range(100):
        rng_key, _ = jax.random.split(rng_key)
        l, c, A, opt_state, nllk_value = step(l, c, A, opt_state, rng_key)

    #     # Debug code
        l_debug_list.append(l)
        c_debug_list.append(c)
        A_debug_list.append(A)
        nll_debug_list.append(nllk_value)
    # # Debug code
    fig = plt.figure(figsize=(15, 6))
    ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    ax_1.plot(l_debug_list)
    ax_2.plot(c_debug_list)
    ax_3.plot(A_debug_list)
    ax_4.plot(nll_debug_list)
    plt.show()

    l, c, A = l, c, A
    K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
    BMC_mean = c * (K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - K_inv.sum() * c * c)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def prior(beta, beta_0, rate, rng_key):
    scale = 1. / rate
    pdf = 1. / scale * jax.scipy.stats.gamma.pdf(beta / scale, a=1 + rate * beta_0)
    return pdf


# @jax.jit
def log_prior(beta, beta_0, rate, rng_key):
    scale = 1. / rate
    logpdf = -jnp.log(scale) + jax.scipy.stats.gamma.logpdf(beta / scale, a=1 + rate * beta_0)
    return logpdf


# @jax.jit
def GP(psi_y_x_mean, psi_y_x_std, X, x_prime, lx):
    """
    :param psi_y_x_mean: n_alpha*1
    :param psi_y_x_std: n_alpha*1
    :param X: n_train*1
    :param x_prime: n_test*1
    :return:
    """
    Nx = psi_y_x_mean.shape[0]
    Mu_standardized, Mu_mean, Mu_std = SIR_utils.standardize(psi_y_x_mean)
    psi_y_x_std = jnp.nan_to_num(psi_y_x_std)
    Sigma_standardized = psi_y_x_std / Mu_std
    X_standardized, X_mean, X_std = SIR_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std
    noise = 1e-2

    K_train_train = my_RBF(X_standardized, X_standardized, lx) + noise * jnp.eye(Nx)  # + jnp.diag(Sigma_standardized)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(x_prime_standardized, X_standardized, lx)
    K_test_test = my_RBF(x_prime_standardized, x_prime_standardized, lx) + noise
    mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
    var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std  # + jnp.mean(psi_y_x_std)

    # plt.figure()
    # plt.plot(x_prime, mu_y_x_prime_original)
    # plt.scatter(X.squeeze(), psi_y_x_mean.squeeze(), color='red')
    # plt.show()
    pause = True
    return mu_y_x_prime_original, std_y_x_prime_original


def peak_infected_number(infections):
    return infections.max(-1)


def peak_infected_time(infections):
    return infections.argmax(-1)


def SIR(args, rng_key):
    Ny_array = jnp.array([10, 20, 50])
    # Ny_array = jnp.arange(2, 60, 2)
    Nx_array = jnp.array([10, 20, 30])
    # N_test = 10
    N_test = 100

    population = float(1e5)
    beta_real, gamma_real = 0.25, 0.05
    beta_0_test = jnp.linspace(0.01, 0.75, N_test)
    rate = 10.0
    scale = 1. / rate
    T = 150
    dt = 1.0
    SamplesNum = 1000

    generate_data_fn = partial(SIR_utils.generate_data, gamma=gamma_real, population=population,
                               T=T, dt=dt, rng_key=rng_key)
    generate_data_vmap = jax.vmap(generate_data_fn)

    if args.mode == 'peak_number':
        f = peak_infected_number
    elif args.mode == 'peak_time':
        f = peak_infected_time
    else:
        raise ValueError('mode should be peak_number or peak_time')

    # Generate ground truth with large number of samples
    ground_truth_array = jnp.zeros([N_test])
    for i in tqdm(range(N_test)):
        beta_0 = beta_0_test[i]
        a = 1 + rate * beta_0
        beta_samples = jax.random.gamma(rng_key, a, shape=(SamplesNum,))
        beta_samples = beta_samples * scale
        temp = generate_data_vmap(beta_samples)
        ground_truth_array = ground_truth_array.at[i].set(f(temp).mean())

    for Nx in Nx_array:
        beta_0_array = jnp.linspace(0.01, 0.75, Nx)

        for Ny in Ny_array:
            BMC_mean_array = jnp.zeros([Nx])
            BMC_std_array = jnp.zeros([Nx])
            MC_mean_array = jnp.zeros([Nx])
            MC_std_array = jnp.zeros([Nx])

            # beta_array_all is Y, beta_0_array is X, f_beta_array_all is f(Y)
            beta_array_all = jnp.zeros([Nx, Ny])
            f_beta_array_all = jnp.zeros([Nx, Ny])

            for j in tqdm(range(Nx)):
                beta_0 = beta_0_array[j]
                a = 1 + rate * beta_0

                rng_key, _ = jax.random.split(rng_key)
                samples = jax.random.gamma(rng_key, a, shape=(SamplesNum, ))
                samples = samples * scale

                rng_key, _ = jax.random.split(rng_key)
                indices = jax.random.permutation(rng_key, jnp.arange(SamplesNum))[:Ny]
                beta_array = samples[indices]

                log_prior_fn = partial(log_prior, beta_0=beta_0, rate=rate, rng_key=rng_key)
                grad_log_prior_fn = jax.grad(log_prior_fn)

                f_beta_array = jnp.zeros([Ny])
                d_log_beta_array = jnp.zeros([Ny, 1])

                for i in range(Ny):
                    beta = beta_array[i]
                    rng_key, _ = jax.random.split(rng_key)
                    D = SIR_utils.generate_data(beta, gamma_real, T, dt, population, rng_key)
                    f_beta = f(D)
                    d_log_beta = grad_log_prior_fn(beta)
                    f_beta_array = f_beta_array.at[i].set(f_beta)
                    d_log_beta_array = d_log_beta_array.at[i, :].set(d_log_beta)

                beta_array_all = beta_array_all.at[j, :].set(beta_array)
                f_beta_array_all = f_beta_array_all.at[j, :].set(f_beta_array)

                MC = Monte_Carlo(f_beta_array)
                f_beta_array_scale, f_beta_array_standardized = SIR_utils.scale(f_beta_array)

                rng_key, _ = jax.random.split(rng_key)
                beta_array_standardized, beta_array_mean, beta_array_std = SIR_utils.standardize(beta_array)

                tt0 = time.time()
                BMC_mean, BMC_std = Bayesian_Monte_Carlo(rng_key, beta_array_standardized[:, None], f_beta_array_standardized,
                                                         d_log_beta_array * beta_array_std, stein_Matern)
                BMC_time_single = time.time() - tt0

                BMC_mean = BMC_mean * f_beta_array_scale
                BMC_std = BMC_std * f_beta_array_scale
                BMC_mean_array = BMC_mean_array.at[j].set(BMC_mean)
                BMC_std_array = BMC_std_array.at[j].set(BMC_std)
                MC_mean_array = MC_mean_array.at[j].set(MC)

                # ========== Debug code ==========
                # large_samples = generate_data_vmap(samples)
                # f_beta_MC_large_sample = f(large_samples).mean()
                # print('True value (MC with large samples)', f_beta_MC_large_sample)
                # print(f'MC with {Ny} number of Y', MC)
                # print(f'BMC with {Ny} number of Y', BMC_mean)
                # print(f"=================")
                # pause = True
                # ========== Debug code ===========

            lx = 1.0
            t0 = time.time()
            BMC_mean, BMC_std = GP(BMC_mean_array[:, None], BMC_std_array[:, None],
                                   beta_0_array[:, None], beta_0_test[:, None], lx)
            BMC_time = time.time() - t0 + BMC_time_single * Nx

            BMC_mean = BMC_mean.squeeze()
            BMC_std = jnp.diag(BMC_std).squeeze()

            # Importance sampling
            t0 = time.time()
            log_py_x_fn = partial(log_prior, rate=rate, rng_key=rng_key)
            IS_mean, _ = SIR_baselines.importance_sampling(log_py_x_fn, beta_0_test[:, None], beta_0_array[:, None], beta_array_all, f_beta_array_all)
            IS_time = time.time() - t0

            # Kernel mean shrinkage estimator
            t0 = time.time()
            KMS_mean, KMS_std = GP(MC_mean_array[:, None], MC_std_array[:, None],
                                   beta_0_array[:, None], beta_0_test[:, None], lx)
            KMS_time = time.time() - t0

            # Least squared Monte Carlo
            t0 = time.time()
            LSMC_mean, _ = SIR_baselines.polynomial(beta_0_array[:, None], beta_array_all[:, None],
                                                    f_beta_array_all, beta_0_test[:, None])
            LSMC_time = time.time() - t0

            SIR_utils.save(args, Nx, Ny, beta_0_test, BMC_mean_array, BMC_mean, BMC_std, KMS_mean,
                           LSMC_mean, IS_mean, ground_truth_array, beta_0_array, BMC_time, KMS_time, LSMC_time, IS_time)

    return


def main(args):
    # seed = int(time.time())
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    SIR(args, rng_key)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/SIR/'
    args.save_path += f"seed_{args.seed}__mode_{args.mode}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path + '/figures/', exist_ok=True)
    return args


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')

    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--mode', type=str, default='peak_number')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
