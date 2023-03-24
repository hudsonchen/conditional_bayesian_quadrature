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


def MCMC(rng_key, beta_0, nsamples, init_params, log_prob, rate):
    rng_key, _ = jax.random.split(rng_key)

    @jax.jit
    def run_chain(rng_key, state):
        num_burnin_steps = int(1e3)
        # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=100,
        #         step_size=1e-2),
        #     num_adaptation_steps=int(num_burnin_steps * 0.8))

        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-4)
        samples = tfp.mcmc.sample_chain(num_results=nsamples,
                                        num_burnin_steps=num_burnin_steps,
                                        current_state=state,
                                        kernel=kernel,
                                        trace_fn=None,
                                        seed=rng_key)
        return samples

    states = run_chain(rng_key, init_params)
    # # # Debug code
    # scale = 1. / rate
    # interval = jnp.linspace(0, 1, 100)
    # interval_pdf = 1. / scale * jax.scipy.stats.gamma.pdf(interval / scale, a=1 + rate * beta_0)
    # plt.figure()
    # plt.plot(interval, interval_pdf)
    # plt.hist(np.array(states), bins=30, alpha=0.8, density=False)
    # plt.show()
    # pause = True
    return states


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
    learning_rate = 5e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    median_d = jnp.median(distance(y, y))
    c_init = c = 1.0
    l_init = l = 2.0
    A_init = A = 1.0
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
    # l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(5000):
        rng_key, _ = jax.random.split(rng_key)
        l, c, A, opt_state, nllk_value = step(l, c, A, opt_state, rng_key)
        # Debug code
        # if jnp.isnan(nllk_value) or jnp.isinf(nllk_value) or jnp.abs(nllk_value) > 1e5:
        #     l, c, A = l, c, A
        #     K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
        #     det = jnp.linalg.det
        #     print(nllk_value)
    #     l_debug_list.append(l)
    #     c_debug_list.append(c)
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # # # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    l, c, A = l, c, A
    K = A * kernel_y(y, y, l, d_log_py, d_log_py) + c + A * jnp.eye(n)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
    BMC_mean = c * (K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - K_inv.sum() * c * c)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def posterior(beta_tilde, beta_mean, beta_std, gamma, D_real, population, beta_0, rate, rng_key):
    scale = 1. / rate
    beta = beta_tilde * beta_std + beta_mean
    prior_beta = jax.scipy.stats.gamma.pdf(beta / scale, a=1 + rate * beta_0)
    S_real, I_real, _, delta_I_real, _ = D_real['S'], D_real['I'], D_real['R'], D_real['dI'], D_real['dR']

    P_sim = 1 - jnp.exp(-beta * (I_real / population))
    part1 = delta_I_real * jnp.log(P_sim)
    part2 = -beta * I_real / population * (S_real - delta_I_real)
    return prior_beta * jnp.exp(part1) * jnp.exp(part2)


# @jax.jit
def log_posterior(beta_tilde, beta_mean, beta_std, gamma, D_real, population, beta_0, rate, rng_key):
    scale = 1. / rate
    beta = beta_tilde * beta_std + beta_mean
    log_prior_beta = jax.scipy.stats.gamma.logpdf(beta / scale, a=1 + rate * beta_0)
    # log_prior_beta = jax.scipy.stats.gamma.logpdf(beta, a=1., loc=beta_0, scale=scale)
    S_real, I_real, _, delta_I_real, _ = D_real['S'], D_real['I'], D_real['R'], D_real['dI'], D_real['dR']

    P_sim = 1 - jnp.exp(-beta * (I_real / population))
    part1 = delta_I_real * jnp.log(P_sim)
    part2 = -beta * I_real / population * (S_real - delta_I_real)
    return (log_prior_beta + (part1 + part2).sum())


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
    noise = 1e-5

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


def peak_infected_number(D):
    return D['dI'].max()


def peak_infected_time(D):
    return D['dI'].argmax()


def SIR(args, rng_key):
    # Ny_list = [2, 3, 5, 7, 10]
    Ny = 10
    population = float(1e5)
    beta_real, gamma_real = 0.25, 0.05
    beta_0_array = jnp.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
    # beta_0_array = jnp.array([0.15, 0.25])
    # N_MCMC = 1000
    N_MCMC = 1000
    N_test = 10
    # N_test = 100
    Nx = len(beta_0_array)
    # beta_0_array = jax.random.uniform(rng_key, shape=(Nx,), minval=0.3, maxval=0.6)
    beta_test_all = jnp.linspace(0.01, 0.60, N_test)
    gamma_lab = 0.05
    rate = 100.0
    scale = 1. / rate
    T = 10
    target_date = 150

    rng_key, _ = jax.random.split(rng_key)
    D_real = SIR_utils.generate_data(beta_real, gamma_real, T, population, rng_key)

    if args.mode == 'peak_number':
        f = peak_infected_number
        rng_key, _ = jax.random.split(rng_key)
        peak_infected_number_array = SIR_utils.ground_truth_peak_infected_number(beta_test_all,
                                                                                 gamma_lab,
                                                                                 D_real,
                                                                                 target_date,
                                                                                 population,
                                                                                 MCMC,
                                                                                 N_MCMC,
                                                                                 log_posterior,
                                                                                 rate,
                                                                                 rng_key)
        jnp.save(f'{args.save_path}/peak_infected_number_array.npy', peak_infected_number_array)
    elif args.mode == 'peak_time':
        f = peak_infected_time
        rng_key, _ = jax.random.split(rng_key)
        peak_infected_time_array = SIR_utils.ground_truth_peak_infected_time(beta_test_all,
                                                                             gamma_lab,
                                                                             D_real,
                                                                             target_date,
                                                                             population,
                                                                             MCMC,
                                                                             N_MCMC,
                                                                             log_posterior,
                                                                             rate,
                                                                             rng_key)
        jnp.save(f'{args.save_path}/peak_infected_time_array.npy', peak_infected_time_array)
    else:
        pass

    bmc_mean_array = jnp.zeros([Nx])
    bmc_std_array = jnp.zeros([Nx])
    mc_mean_array = jnp.zeros([Nx])
    mc_std_array = jnp.zeros([Nx])

    # beta_array_all is Y, beta_0_array is X, f_beta_array_all is f(Y)
    beta_array_all = jnp.zeros([Nx, Ny])
    f_beta_array_all = jnp.zeros([Nx, Ny])

    for j in tqdm(range(Nx)):
        beta_0 = beta_0_array[j]
        init_params = beta_0
        rng_key, _ = jax.random.split(rng_key)
        log_posterior_fn = partial(log_posterior, beta_mean=0., beta_std=1.0, gamma=gamma_lab, D_real=D_real,
                                   population=population, beta_0=beta_0, rate=rate, rng_key=rng_key)
        rng_key, _ = jax.random.split(rng_key)
        samples_post = MCMC(rng_key, beta_0, N_MCMC, init_params, log_posterior_fn, rate)
        samples_post = jnp.unique(samples_post, axis=0)
        rng_key, _ = jax.random.split(rng_key)
        samples_post = jax.random.permutation(rng_key, samples_post)

        # Debug : Large sample Monte Carlo
        beta_array_large_sample = jnp.zeros([N_MCMC, 1])
        f_beta_array_large_sample = jnp.zeros([N_MCMC, 1])
        print(f"Sampling {N_MCMC} with MCMC to estimate the true value")
        for i in range(N_MCMC):
            beta = samples_post[i]
            rng_key, _ = jax.random.split(rng_key)
            D = SIR_utils.generate_data(beta, gamma_real, target_date, population, rng_key)
            f_beta = f(D)
            beta_array_large_sample = beta_array_large_sample.at[i, :].set(beta)
            f_beta_array_large_sample = f_beta_array_large_sample.at[i, :].set(f_beta)
        MC_large_sample = f_beta_array_large_sample.mean()

        rng_key, _ = jax.random.split(rng_key)
        samples_post = jax.random.permutation(rng_key, samples_post)
        beta_array = samples_post[:Ny]
        beta_standardized, beta_mean, beta_std = SIR_utils.standardize(beta_array)
        log_posterior_fn = partial(log_posterior, beta_mean=beta_mean, beta_std=beta_std,
                                   gamma=gamma_lab, D_real=D_real,
                                   population=population, beta_0=beta_0,
                                   rate=rate, rng_key=rng_key)
        grad_log_posterior_fn = jax.grad(log_posterior_fn)

        f_beta_array = jnp.zeros([Ny])
        d_log_beta_array = jnp.zeros([Ny, 1])
        for i in range(Ny):
            rng_key, _ = jax.random.split(rng_key)
            D = SIR_utils.generate_data(beta_array[i], gamma_real, target_date, population, rng_key)
            f_beta = f(D)
            d_log_beta = grad_log_posterior_fn(beta_standardized[i])
            f_beta_array = f_beta_array.at[i].set(f_beta)
            d_log_beta_array = d_log_beta_array.at[i, :].set(d_log_beta)

        beta_array_all = beta_array_all.at[j, :].set(beta_array)
        f_beta_array_all = f_beta_array_all.at[j, :].set(f_beta_array)

        MC = Monte_Carlo(f_beta_array)
        f_beta_array_scale, f_beta_array_standardized = SIR_utils.scale(f_beta_array)
        beta_standardized = beta_standardized[:, None]
        rng_key, _ = jax.random.split(rng_key)

        bmc_mean, bmc_std = Bayesian_Monte_Carlo(rng_key, beta_standardized, f_beta_array_standardized,
                                                 d_log_beta_array, stein_Matern)
        bmc_mean = bmc_mean * f_beta_array_scale
        bmc_std = bmc_std * f_beta_array_scale
        bmc_mean_array = bmc_mean_array.at[j].set(bmc_mean)
        bmc_std_array = bmc_std_array.at[j].set(bmc_std)
        mc_mean_array = mc_mean_array.at[j].set(MC)
        # Debug
        print('True value (MC with large samples)', MC_large_sample)
        print(f'MC with {Ny} number of Y', MC)
        print(f'BMC with {Ny} number of Y', bmc_mean)
        print(f"=================")
        pause = True

    lx = 1.0
    BMC_mean, BMC_std = GP(bmc_mean_array[:, None], bmc_std_array[:, None],
                           beta_0_array[:, None], beta_test_all[:, None], lx)
    BMC_mean = BMC_mean.squeeze()
    BMC_std = jnp.diag(BMC_std).squeeze()

    # Importance sampling
    # py_x_fn = partial(posterior, beta_mean=0., beta_std=1., gamma=gamma_lab, D_real=D_real,
    #                   population=population, rate=rate, rng_key=rng_key)
    # IS_mean, _ = SIR_baselines.importance_sampling(py_x_fn, beta_test_all, beta_0_array, beta_array_all, f_beta_array_all)

    # Kernel mean shrinkage estimator
    KMS_mean, KMS_std = GP(mc_mean_array[:, None], mc_std_array[:, None],
                           beta_0_array[:, None], beta_test_all[:, None], lx)

    # Least squared Monte Carlo
    poly_mean, _ = SIR_baselines.polynomial(beta_0_array[:, None], beta_array_all[:, None],
                                            f_beta_array_all, beta_test_all[:, None])

    jnp.save(f"{args.save_path}/BMC_mean.npy", BMC_mean.squeeze())
    jnp.save(f"{args.save_path}/BMC_std.npy", BMC_std.squeeze())
    jnp.save(f"{args.save_path}/KMS_mean.npy", KMS_mean.squeeze())
    jnp.save(f"{args.save_path}/poly_mean.npy", poly_mean.squeeze())

    plt.figure()
    plt.plot(beta_test_all, BMC_mean, color='blue', label='BMC')
    plt.plot(beta_test_all, KMS_mean, color='red', label='KMS')
    plt.plot(beta_test_all, poly_mean, color='green', label='LSMC')
    if args.mode == 'peak_number':
        plt.plot(beta_test_all, jnp.load(f'{args.save_path}/peak_infected_number_array.npy'), color='black',
                 label='True')
    elif args.mode == 'peak_time':
        plt.plot(beta_test_all, jnp.load(f'{args.save_path}/peak_infected_time_array.npy'), color='black', label='True')
    else:
        pass
    plt.scatter(beta_0_array, bmc_mean_array, color='orange')
    plt.fill_between(beta_test_all, BMC_mean - BMC_std, BMC_mean + BMC_std, alpha=0.2, color='blue')
    plt.legend()
    plt.savefig(f"{args.save_path}/plot.pdf")
    plt.show()
    pause = True
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
