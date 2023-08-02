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
import baselines
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
    # os.chdir("/home/zongchen/CBQ")
    os.chdir("/home/zongchen/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'ucabzc9':
    os.chdir("/home/ucabzc9/Scratch/CBQ")
else:
    pass


def Monte_Carlo(fx):
    return fx.mean(0)


# @partial(jax.jit, static_argnums=(4,))
def Bayesian_Monte_Carlo(rng_key, x, fx, d_log_px, kernel_x):
    """
    :param rng_key:
    :param x: N * D
    :param fx: N
    :param d_log_px: N * D
    :param kernel_x: kernel function
    :return:
    """
    n = x.shape[0]
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    c_init = c = 1.0
    l_init = l = 1.5
    A_init = A = 1.0 / jnp.sqrt(n)

    # @jax.jit
    # def nllk_func(l, c, A):
    #     l, c, A = l, c, A
    #     n = y.shape[0]
    #     K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
    #     K_inv = jnp.linalg.inv(K)
    #     nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
    #     return nll
    #
    # @jax.jit
    # def step(l, c, A, opt_state, rng_key):
    #     nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(l, c, A)
    #     updates, opt_state = optimizer.update(grads, opt_state, (l, c, A))
    #     l, c, A = optax.apply_updates((l, c, A), updates)
    #     return l, c, A, opt_state, nllk_value

    # ========== Debug code ==========
    # l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    # ========== Debug code ==========

    # for _ in range(0):
    #     rng_key, _ = jax.random.split(rng_key)
    #     l, c, A, opt_state, nllk_value = step(l, c, A, opt_state, rng_key)
    #     if jnp.isnan(nllk_value):
    #         break
    # ========== Debug code ==========
    #     l_debug_list.append(l)
    #     c_debug_list.append(c)
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()
    # ========== Debug code ==========

    l, c, A = l, c, A
    K = A * kernel_x(x, x, l, d_log_px, d_log_px) + c + A * jnp.eye(n)
    K_inv = jnp.linalg.inv(K)
    I_BQ_mean = c * (K_inv @ fx).sum()
    I_BQ_std = jnp.sqrt(c - K_inv.sum() * c * c)
    pause = True
    return I_BQ_mean, I_BQ_std


# @jax.jit
def prior(X, theta, rate, rng_key):
    scale = 1. / rate
    pdf = 1. / scale * jax.scipy.stats.gamma.pdf(X / scale, a=1 + rate * theta)
    return pdf


# @jax.jit
def log_prior(X, theta, rate, rng_key):
    scale = 1. / rate
    logpdf = -jnp.log(scale) + jax.scipy.stats.gamma.logpdf(X / scale, a=1 + rate * theta)
    return logpdf


# @jax.jit
def GP(psi_y_x_mean, psi_y_x_std, X, x_prime, ground_truth):
    """
    :param psi_y_x_mean: n_alpha*1
    :param psi_y_x_std: n_alpha*1
    :param X: n_train*1
    :param x_prime: n_test*1
    :return:
    """
    eps = 1e-6
    T = psi_y_x_mean.shape[0]
    Mu_standardized, Mu_mean, Mu_std = SIR_utils.standardize(psi_y_x_mean)
    X_standardized, X_mean, X_std = SIR_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std

    if psi_y_x_std is None:
        l_array = jnp.array([0.3, 1.0, 3.0])
        sigma_array = jnp.array([0.1, 0.01, 0.001])
        nll_array = jnp.zeros([l_array.shape[0], sigma_array.shape[0]]) + 0.0
        A_array = jnp.zeros([l_array.shape[0], sigma_array.shape[0]]) + 0.0

        for i, l in enumerate(l_array):
            for j, sigma in enumerate(sigma_array):
                K_no_scale = my_Matern(X_standardized, X_standardized, l)
                A = Mu_standardized.T @ K_no_scale @ Mu_standardized / T
                A_array = A_array.at[i, j].set(A[0][0])
                K = A * K_no_scale
                K_inv = jnp.linalg.inv(K + sigma * jnp.eye(T))
                nll = -(-0.5 * Mu_standardized.T @ K_inv @ Mu_standardized - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / T
                nll_array = nll_array.at[i].set(nll[0][0])
        min_index_flat = jnp.argmin(nll_array)
        i1, i2 = jnp.unravel_index(min_index_flat, nll_array.shape)
        l = l_array[i1]
        sigma = sigma_array[i2]
        A = A_array[i1, i2]

        K_train_train = A * my_Matern(X_standardized, X_standardized, l) + sigma * jnp.eye(T)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * my_Matern(x_prime_standardized, X_standardized, l)
        K_test_test = A * my_Matern(x_prime_standardized, x_prime_standardized, l) + sigma
        mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
        var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
        std_y_x_prime = jnp.sqrt(var_y_x_prime)

    else:
        l_array = jnp.array([5.0])
        nll_array = 0.0 * l_array
        A_array = 0.0 * l_array
        sigma = psi_y_x_std / Mu_std
        for i, l in enumerate(l_array):
            K_no_scale = my_Matern(X_standardized, X_standardized, l)
            A = Mu_standardized.T @ K_no_scale @ Mu_standardized / T
            A_array = A_array.at[i].set(A[0][0])
            K = A * K_no_scale
            K_inv = jnp.linalg.inv(K + jnp.diag(sigma ** 2))
            nll = -(-0.5 * Mu_standardized.T @ K_inv @ Mu_standardized - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / T
            nll_array = nll_array.at[i].set(nll[0][0])
        l = l_array[nll_array.argmin()]
        A = A_array[nll_array.argmin()]

        if T > 10:
            sigma = jnp.ones_like(Mu_standardized) * 0.1
        else:
            pass
        K_train_train = A * my_Matern(X_standardized, X_standardized, l) + jnp.diag(sigma ** 2)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = A * my_Matern(x_prime_standardized, X_standardized, l)
        K_test_test = A * my_Matern(x_prime_standardized, x_prime_standardized, l) + (sigma ** 2).mean()
        mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
        var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
        std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std
    # ========== Debug code ==========
    # plt.figure()
    # plt.plot(x_prime.squeeze(), ground_truth, color='black')
    # plt.plot(x_prime.squeeze(), mu_y_x_prime_original.squeeze(), color='red')
    # plt.scatter(X.squeeze(), psi_y_x_mean.squeeze(), color='blue')
    # plt.show()
    # pause = True
    # ========== Debug code ==========
    return mu_y_x_prime_original, std_y_x_prime_original


def peak_infected_number(infections):
    return infections.max(-1)


def SIR(args, rng_key):
    N_array = jnp.array([10, 20, 30])
    # N_array = jnp.arange(5, 45, 5)
    T_array = jnp.array([10])
    # T_array = jnp.arange(5, 45, 5)
    # T_test = 10
    T_test = 10

    population = float(1e5)
    X_real, gamma_real = 0.25, 0.05
    theta_test = jnp.linspace(0.1, 0.8, T_test)
    rate = 10.0
    scale = 1. / rate
    Time = 150
    dt = 1.0
    SamplesNum = 10000

    generate_data_fn = partial(SIR_utils.generate_data, gamma=gamma_real, population=population,
                               T=Time, dt=dt, rng_key=rng_key)
    generate_data_vmap = jax.vmap(generate_data_fn)

    f = peak_infected_number

    # Generate ground truth with large number of samples
    ground_truth_array = jnp.zeros([T_test])
    for i in tqdm(range(T_test)):
        theta = theta_test[i]
        a = 1 + rate * theta
        rng_key, _ = jax.random.split(rng_key)
        X_samples = jax.random.gamma(rng_key, a, shape=(SamplesNum,))
        X_samples = X_samples * scale
        temp = generate_data_vmap(X_samples)
        ground_truth_array = ground_truth_array.at[i].set(f(temp).mean())

    for T in T_array:
        theta_array = jnp.linspace(0.1, 0.8, T)

        for N in N_array:
            I_BQ_mean_array = jnp.zeros([T])
            I_BQ_std_array = jnp.zeros([T])
            I_MC_mean_array = jnp.zeros([T])
            I_MC_std_array = jnp.zeros([T])

            X_array_all = jnp.zeros([T, N])
            f_X_array_all = jnp.zeros([T, N])

            for j in tqdm(range(T)):
                theta = theta_array[j]
                a = 1 + rate * theta

                rng_key, _ = jax.random.split(rng_key)
                samples = jax.random.gamma(rng_key, a, shape=(SamplesNum,))
                samples = samples * scale
                rng_key, _ = jax.random.split(rng_key)
                indices = jax.random.permutation(rng_key, jnp.arange(SamplesNum))[:N]
                X_array = samples[indices]

                log_prior_fn = partial(log_prior, theta=theta, rate=rate, rng_key=rng_key)
                grad_log_prior_fn = jax.grad(log_prior_fn)

                f_X_array = jnp.zeros([N])
                d_log_X_array = jnp.zeros([N, 1])

                for i in range(N):
                    X = X_array[i]
                    rng_key, _ = jax.random.split(rng_key)
                    D = SIR_utils.generate_data(X, gamma_real, Time, dt, population, rng_key)
                    f_X = f(D)
                    d_log_X = grad_log_prior_fn(X)
                    f_X_array = f_X_array.at[i].set(f_X)
                    d_log_X_array = d_log_X_array.at[i, :].set(d_log_X)

                X_array_all = X_array_all.at[j, :].set(X_array)
                f_X_array_all = f_X_array_all.at[j, :].set(f_X_array)

                I_MC = Monte_Carlo(f_X_array)
                f_X_array_scale, f_X_array_standardized = SIR_utils.scale(f_X_array)

                rng_key, _ = jax.random.split(rng_key)
                X_array_standardized, X_array_mean, X_array_std = SIR_utils.standardize(X_array)

                _, _ = Bayesian_Monte_Carlo(rng_key, X_array_standardized[:, None],
                                            f_X_array_standardized,
                                            d_log_X_array * X_array_std, stein_Matern)
                tt0 = time.time()
                I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo(rng_key, X_array_standardized[:, None],
                                                          f_X_array_standardized,
                                                          d_log_X_array * X_array_std, stein_Matern)
                tt1 = time.time()

                I_BQ_mean = I_BQ_mean * f_X_array_scale
                I_BQ_std = I_BQ_std * f_X_array_scale
                if I_BQ_mean > 2 * I_MC:
                    I_BQ_mean = I_MC
                I_BQ_mean_array = I_BQ_mean_array.at[j].set(I_BQ_mean)
                I_BQ_std_array = I_BQ_std_array.at[j].set(I_BQ_std)
                I_MC_mean_array = I_MC_mean_array.at[j].set(I_MC)

                # ========== Debug code ==========
                # large_samples = generate_data_vmap(samples)
                # f_X_MC_large_sample = f(large_samples).mean()
                # print(f'True value (MC with {SamplesNum} samples)', f_X_MC_large_sample)
                # print(f'MC with {N} number of Y', I_MC)
                # print(f'BMC with {N} number of Y', I_BQ_mean)
                # print(f"=================")
                # pause = True
                # ========== Debug code ===========

            I_BQ_std_array = jnp.nan_to_num(I_BQ_std_array, nan=0.)
            I_BQ_std_array = jnp.ones_like(I_BQ_std_array) * jnp.mean(I_BQ_std_array)

            _, _ = GP(I_BQ_mean_array[:, None], I_BQ_std_array[:, None],
                      theta_array[:, None], theta_test[:, None], ground_truth_array)
            t0 = time.time()
            I_BQ_mean, I_BQ_std = GP(I_BQ_mean_array[:, None], I_BQ_std_array[:, None],
                                   theta_array[:, None], theta_test[:, None], ground_truth_array)
            BMC_time = time.time() - t0 + (tt1 - tt0) * T

            I_BQ_mean = I_BQ_mean.squeeze()
            I_BQ_std = jnp.diag(I_BQ_std.squeeze())

            # Importance sampling
            log_px_theta_fn = partial(log_prior, rate=rate, rng_key=rng_key)
            _, _ = baselines.importance_sampling_SIR(log_px_theta_fn, theta_test[:, None], theta_array[:, None],
                                                     X_array_all, f_X_array_all)
            t0 = time.time()
            IS_mean, _ = baselines.importance_sampling_SIR(log_px_theta_fn, theta_test[:, None], theta_array[:, None],
                                                           X_array_all, f_X_array_all)
            IS_time = time.time() - t0

            # Kernel mean shrinkage estimator
            _, _ = GP(I_MC_mean_array[:, None], None,
                      theta_array[:, None], theta_test[:, None], ground_truth_array)
            t0 = time.time()
            KMS_mean, KMS_std = GP(I_MC_mean_array[:, None], None,
                                   theta_array[:, None], theta_test[:, None], ground_truth_array)
            KMS_time = time.time() - t0

            # Least squared Monte Carlo
            _, _ = baselines.polynomial(theta_array[:, None], X_array_all[:, None],
                                            f_X_array_all, theta_test[:, None])
            t0 = time.time()
            LSMC_mean, _ = baselines.polynomial(theta_array[:, None], X_array_all[:, None],
                                                    f_X_array_all, theta_test[:, None])
            LSMC_time = time.time() - t0

            calibration = SIR_utils.calibrate(ground_truth_array, I_BQ_mean, I_BQ_std)

            SIR_utils.save(args, T, N, theta_test, I_BQ_mean_array, I_BQ_mean, I_BQ_std, KMS_mean,
                           LSMC_mean, IS_mean, ground_truth_array, theta_array, BMC_time, KMS_time, LSMC_time, IS_time,
                           calibration)

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
    print(f'seed is {args.seed}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
