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
import argparse
import os
import pwd

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


def MCMC(rng_key, beta_lab, nsamples, init_params, log_prob, rate):
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

        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-3)
        samples = tfp.mcmc.sample_chain(num_results=nsamples,
                                        num_burnin_steps=num_burnin_steps,
                                        current_state=state,
                                        kernel=kernel,
                                        trace_fn=None,
                                        seed=rng_key)
        return samples

    states = run_chain(rng_key, init_params)
    # # Debug code
    # scale = 1. / rate
    # interval = jnp.linspace(0, 1, 100)
    # interval_pdf = 1. / scale * jax.scipy.stats.gamma.pdf(interval / scale, a=1 + rate * beta_lab)
    # plt.figure()
    # plt.plot(interval, interval_pdf)
    # plt.hist(np.array(states), bins=20, alpha=0.8, density=True)
    # plt.show()
    pause = True
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
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    median_d = jnp.median(distance(y, y))
    c_init = c = 10.0
    log_l_init = log_l = jnp.log(0.01)
    log_A_init = log_A = jnp.log(1.0)
    A_extra_scale = 1e-5
    opt_state = optimizer.init((log_l_init, c_init, log_A_init))

    @jax.jit
    def nllk_func(log_l, c, log_A):
        l, c, A = jnp.exp(log_l), c, jnp.exp(log_A)
        n = y.shape[0]
        K = A * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c + jnp.eye(n)
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
        return nll

    @jax.jit
    def step(log_l, c, log_A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, log_A)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, log_A))
        log_l, c, log_A = optax.apply_updates((log_l, c, log_A), updates)
        return log_l, c, log_A, opt_state, nllk_value

    # # Debug code
    # log_l_debug_list = []
    # c_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(10000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, log_A, opt_state, nllk_value = step(log_l, c, log_A, opt_state, rng_key)
        # Debug code
    #     if jnp.isnan(nllk_value) or jnp.isinf(nllk_value) or jnp.abs(nllk_value) > 1e5:
    #         l, c, A = jnp.exp(log_l), c, jnp.exp(log_A)
    #         K = A * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c + jnp.eye(n)
    #         det = jnp.linalg.det
    #         print(nllk_value)
    #     log_l_debug_list.append(log_l)
    #     c_debug_list.append(c)
    #     A_debug_list.append(jnp.exp(log_A))
    #     nll_debug_list.append(nllk_value)
    # # # Debug code
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    # ax_1.plot(log_l_debug_list)
    # ax_2.plot(c_debug_list)
    # ax_3.plot(A_debug_list)
    # ax_4.plot(nll_debug_list)
    # plt.show()

    l, c, A = jnp.exp(log_l), c, jnp.exp(log_A)
    final_K = A * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c + jnp.eye(n)
    final_K_inv = jnp.linalg.inv(final_K + eps * jnp.eye(n))
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def log_posterior(beta, gamma, D_real, population, beta_lab, T, log_posterior_scale, rate, rng_key):
    scale = 1. / rate

    log_prior_beta = jax.scipy.stats.gamma.logpdf(beta / scale, a=1 + rate * beta_lab)
    # log_prior_beta = jax.scipy.stats.gamma.logpdf(beta, a=1., loc=beta_lab, scale=scale)
    S_real, I_real, _, delta_I_real, _ = D_real['S'], D_real['I'], D_real['R'], D_real['dI'], D_real['dR']

    P_sim = 1 - jnp.exp(-beta * (I_real / population))
    part1 = delta_I_real * jnp.log(P_sim)
    part2 = -beta * I_real / population * (S_real - delta_I_real)
    # The scale is used to make MCMC stable
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

    K_train_train = my_RBF(X_standardized, X_standardized, lx) + jnp.diag(Sigma_standardized) \
                    + noise * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(x_prime_standardized, X_standardized, lx)
    K_test_test = my_RBF(x_prime_standardized, x_prime_standardized, lx) + noise
    mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
    var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std + jnp.mean(psi_y_x_std)

    plt.figure()
    plt.plot(x_prime, mu_y_x_prime_original)
    plt.show()
    pause = True
    return mu_y_x_prime_original, std_y_x_prime_original


def peak_infected_number(D):
    return D['I'].max()


def peak_infected_time(D):
    return D['I'].argmax()


def SIR(args, rng_key):
    # Ny_list = [2, 3, 5, 7, 10]
    Ny_list = [20]
    population = float(1e5)
    beta_real, gamma_real = 0.15, 0.05
    beta_lab_array = jnp.array([0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.25, 0.35, 0.45])
    # beta_lab_array = jnp.array([0.05, 0.10, 0.15, 0.25, 0.35])
    Nx = len(beta_lab_array)
    # beta_lab_array = jax.random.uniform(rng_key, shape=(Nx,), minval=0.3, maxval=0.6)
    beta_lab_all = jnp.linspace(0.01, 0.45, 50)
    gamma_lab = 0.05
    rate = 1000.0
    scale = 1. / rate
    T = 150

    target_date = 20

    rng_key, _ = jax.random.split(rng_key)
    D_real, D_real_target = SIR_utils.generate_data(beta_real, gamma_real, T, population, target_date, rng_key)
    D_real_target = SIR_utils.convert_dict_to_jnp(D_real_target)
    N_MCMC = 100

    if args.mode == 'peak_number':
        f = peak_infected_number
        rng_key, _ = jax.random.split(rng_key)
        peak_infected_number_array = SIR_utils.ground_truth_peak_infected_number(beta_lab_all,
                                                                                 gamma_lab, T, population,
                                                                                 target_date, rng_key)
        jnp.save(f'./data/SIR/peak_infected_number_array.npy', peak_infected_number_array)
    elif args.mode == 'peak_time':
        f = peak_infected_time
        rng_key, _ = jax.random.split(rng_key)
        peak_infected_time_array = SIR_utils.ground_truth_peak_infected_time(beta_lab_all,
                                                                             gamma_lab, T, population,
                                                                             target_date, rng_key)
        jnp.save(f'./data/SIR/peak_infected_time_array.npy', peak_infected_time_array)
    else:
        pass

    psi_mean_array = jnp.zeros([Nx])
    psi_std_array = jnp.zeros([Nx])

    for j in tqdm(range(Nx)):
        beta_lab = beta_lab_array[j]
        init_params = beta_lab
        # This one is heuristic
        log_posterior_scale = 100.
        rng_key, _ = jax.random.split(rng_key)
        log_posterior_fn = partial(log_posterior, gamma=gamma_lab, D_real=D_real_target,
                                   population=population, beta_lab=beta_lab, T=T,
                                   log_posterior_scale=log_posterior_scale, rate=rate, rng_key=rng_key)
        grad_log_posterior_fn = jax.grad(log_posterior_fn)
        rng_key, _ = jax.random.split(rng_key)
        samples_post = MCMC(rng_key, beta_lab, N_MCMC, init_params, log_posterior_fn, rate)
        samples_post = jnp.unique(samples_post, axis=0)
        rng_key, _ = jax.random.split(rng_key)
        samples_post = jax.random.permutation(rng_key, samples_post)

        # Debug : Large sample Monte Carlo
        beta_array_large_sample = jnp.zeros([N_MCMC, 1])
        f_beta_array_large_sample = jnp.zeros([N_MCMC, 1])
        print(f"Sampling {N_MCMC} with MCMC to estimate the true value")
        for i in tqdm(range(N_MCMC)):
            beta = samples_post[i]
            rng_key, _ = jax.random.split(rng_key)
            D, _ = SIR_utils.generate_data(beta, gamma_real, T, population, target_date, rng_key)
            f_beta = f(D)
            beta_array_large_sample = beta_array_large_sample.at[i, :].set(beta)
            f_beta_array_large_sample = f_beta_array_large_sample.at[i, :].set(f_beta)
        MC_large_sample = f_beta_array_large_sample.mean()

        for Ny in Ny_list:
            beta_array = jnp.zeros([Ny, 1])
            f_beta_array = jnp.zeros([Ny])
            d_log_beta_array = jnp.zeros([Ny, 1])
            for i in range(Ny):
                beta = samples_post[i]
                rng_key, _ = jax.random.split(rng_key)
                D, _ = SIR_utils.generate_data(beta, gamma_real, T, population, target_date, rng_key)
                f_beta = f(D)
                d_log_beta = grad_log_posterior_fn(beta)
                beta_array = beta_array.at[i, :].set(beta)
                f_beta_array = f_beta_array.at[i].set(f_beta)
                d_log_beta_array = d_log_beta_array.at[i, :].set(d_log_beta)

            MC = Monte_Carlo(f_beta_array)
            f_beta_array_standardized, f_beta_array_mean, f_beta_array_std = SIR_utils.standardize(f_beta_array)
            rng_key, _ = jax.random.split(rng_key)
            psi_mean, psi_std = Bayesian_Monte_Carlo(rng_key, beta_array, f_beta_array_standardized, d_log_beta_array,
                                                     stein_Gaussian)
            psi_mean = psi_mean * f_beta_array_std + f_beta_array_mean
            psi_std = psi_std * f_beta_array_std
            psi_mean_array = psi_mean_array.at[j].set(psi_mean)
            psi_std_array = psi_std_array.at[j].set(psi_std)

            # Debug
            print('True value (MC with large samples)', MC_large_sample)
            print(f'MC with {Ny} number of Y', MC)
            print(f'BMC with {Ny} number of Y', psi_mean)
            print(f"=================")
            pause = True
    if args.mode == 'peak_number':
        # plt.plot(beta_lab_all, jnp.load(f'./data/SIR/peak_infected_number_array.npy'))
        lx = 2.0
    elif args.mode == 'peak_time':
        # plt.plot(beta_lab_all, jnp.load(f'./data/SIR/peak_infected_time_array.npy'))
        lx = 0.7
    else:
        pass
    BMC_mean, BMC_std = GP(psi_mean_array[:, None], psi_std_array[:, None],
                           beta_lab_array[:, None], beta_lab_all[:, None], lx)
    BMC_mean = BMC_mean.squeeze()
    BMC_std = jnp.diag(BMC_std).squeeze()
    plt.figure()
    plt.plot(beta_lab_all, BMC_mean, color='blue')
    if args.mode == 'peak_number':
        # plt.plot(beta_lab_all, jnp.load(f'./data/SIR/peak_infected_number_array.npy'))
        plt.axhline(D_real['I'].max(), linestyle='--', color='black')
    elif args.mode == 'peak_time':
        # plt.plot(beta_lab_all, jnp.load(f'./data/SIR/peak_infected_time_array.npy'))
        plt.axhline(D_real['I'].argmax(), linestyle='--', color='black')
    else:
        pass
    plt.scatter(beta_lab_array, psi_mean_array, color='orange')
    plt.fill_between(beta_lab_all, BMC_mean - BMC_std, BMC_mean + BMC_std, alpha=0.4, color='blue')
    plt.savefig(f"{args.save_path}/{args.mode}.pdf")
    plt.show()
    pause = True
    return


def main(args):
    # seed = int(time.time())
    seed = 1
    rng_key = jax.random.PRNGKey(seed)
    SIR(args, rng_key)
    return


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')

    # Args settings
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./results/SIR')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--mode', type=str, default='peak_number')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_config()
    main(args)