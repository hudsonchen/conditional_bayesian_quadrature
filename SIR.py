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
tfd = tfp.distributions
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

@jax.jit
def time_step(beta, gamma, population, St, It, Rt, rng_key):
    Pt = It / population
    rng_key, _ = jax.random.split(rng_key)
    dist = tfd.Binomial(total_count=St, probs=1 - jnp.exp(-beta * Pt))
    delta_It = dist.sample(seed=rng_key)
    rng_key, _ = jax.random.split(rng_key)
    dist = tfd.Binomial(total_count=It, probs=gamma)
    delta_Rt = dist.sample(seed=rng_key)
    St = St - delta_It
    It = It + delta_It - delta_Rt
    Rt = Rt + delta_Rt
    return St, It, Rt, delta_It, delta_Rt


def generate_data(beta, gamma, T, population, target_date, rng_key):
    """
    :param beta: float, infection rate
    :param gamma: float, recovery rate
    :param rng_key:
    :param T: Time length
    :param target_date: The time for the second retuned result
    :return: array T*3, the first is number of Susceptible,
    the second is Infected, the third is Recoverdd
    """
    It, Rt = 50., 0.
    St = population - It - Rt

    S_list = []
    I_list = []
    R_list = []
    delta_It_list = []
    delta_Rt_list = []
    # Note the index here
    # S_0 to S_{T-1}, I_0 to I_{T-1}, R_0 to R_{T-1}
    # delta_I1 to delta_IT, delta_R1 to delta_RT
    for i in range(T):
        S_list.append(St)
        I_list.append(It)
        R_list.append(Rt)
        St, Rt, It, delta_It, delta_Rt = time_step(beta, gamma, population, St, It, Rt, rng_key)
        delta_It_list.append(delta_It)
        delta_Rt_list.append(delta_Rt)

    S_array = np.array(S_list)
    I_array = np.array(I_list)
    R_array = np.array(R_list)
    delta_It_array = np.array(delta_It_list)
    delta_Rt_array = np.array(delta_Rt_list)

    # # Plot the data on three separate curves for S(t), I(t) and R(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    # ax.plot(np.arange(T), S_array, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(np.arange(T), I_array, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.plot(np.arange(T), R_array, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    # ax.set_xlabel('Time /days')
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # plt.show()

    D_real = {'S': S_array, 'I': I_array, 'R': R_array, 'dI': delta_It_array, 'dR': delta_Rt_array}
    # non_zero_ind = SIR_utils.non_zero_ind(delta_It_array)
    # St_real_non_zero = S_array[non_zero_ind]
    # It_real_non_zero = I_array[non_zero_ind]
    # Rt_real_non_zero = R_array[non_zero_ind]
    # delta_It_real_non_zero = delta_It_array[non_zero_ind]
    # delta_Rt_real_non_zero = delta_Rt_array[non_zero_ind]
    # D_real_remove_zero = {'S': St_real_non_zero, 'I': It_real_non_zero,
    #                       'R': Rt_real_non_zero, 'dI': delta_It_real_non_zero,
    #                       'dR': delta_Rt_real_non_zero}
    D_real_remove_zero = {'S': S_array[:target_date], 'I': I_array[:target_date],
                          'R': R_array[:target_date], 'dI': delta_It_array[:target_date],
                          'dR': delta_Rt_array[:target_date]}
    return D_real, D_real_remove_zero


def MCMC(rng_key, beta_lab, nsamples, init_params, log_prob):
    rng_key, _ = jax.random.split(rng_key)
    @jax.jit
    def run_chain(rng_key, state):
        num_burnin_steps = int(1e3)
        # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=3,
        #         step_size=1.0),
        #         num_adaptation_steps=int(num_burnin_steps * 0.8))

        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-2)
        samples = tfp.mcmc.sample_chain(num_results=nsamples,
                                        num_burnin_steps=num_burnin_steps,
                                        current_state=state,
                                        kernel=kernel,
                                        trace_fn=None,
                                        seed=rng_key)
        return samples

    states = run_chain(rng_key, init_params)
    # Debug code
    # rate = 10.0
    # scale = 1. / rate
    # interval = jnp.linspace(0, 1, 100)
    # interval_pdf = 1. / scale * jax.scipy.stats.gamma.pdf(interval / scale, a=rate * beta_lab)
    # plt.figure()
    # plt.plot(interval, interval_pdf)
    # plt.hist(np.array(states), bins=10, alpha=0.8, density=True)
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
    N, D = y.shape[0], y.shape[1]
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6
    median_d = jnp.median(distance(y, y))
    c_init = c = 1.0
    log_l_init = log_l = jnp.log(0.03)
    A_init = A = 1.0
    A_extra_scale = 1.0
    opt_state = optimizer.init((log_l_init, c_init, A_init))
    d_log_py /= 1e3

    @jax.jit
    def nllk_func(log_l, c, A):
        l, c, A = jnp.exp(log_l), c, A
        n = y.shape[0]
        K = A_init * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c
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
    log_l_debug_list = []
    c_debug_list = []
    A_debug_list = []
    nll_debug_list = []
    for _ in range(10000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # # Debug code
        if jnp.isnan(nllk_value) or jnp.isinf(nllk_value) or jnp.abs(nllk_value) > 1e5:
            l = jnp.exp(log_l)
            l = jnp.exp(log_l_debug_list[-1])
            A = A_debug_list[-1]
            c = c_debug_list[-1]
            K = A * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c
            print(nllk_value)
        log_l_debug_list.append(log_l)
        c_debug_list.append(c)
        A_debug_list.append(A)
        nll_debug_list.append(nllk_value)
    # Debug code
    fig = plt.figure(figsize=(15, 6))
    ax_1, ax_2, ax_3, ax_4 = fig.subplots(1, 4)
    ax_1.plot(log_l_debug_list)
    ax_2.plot(c_debug_list)
    ax_3.plot(A_debug_list)
    ax_4.plot(nll_debug_list)
    plt.show()

    l, c, A = jnp.exp(log_l), c, A
    final_K = A * kernel_y(y, y, l, d_log_py, d_log_py) * A_extra_scale + c
    final_K_inv = jnp.linalg.inv(final_K + eps * jnp.eye(n))
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def log_posterior(beta, gamma, D_real, population, beta_lab, T, log_posterior_scale, rng_key):
    rate = 10.
    scale = 1. / rate

    log_prior_beta = jax.scipy.stats.gamma.logpdf(beta / scale, a=rate * beta_lab)
    S_real, I_real, _, delta_I_real, _ = D_real['S'], D_real['I'], D_real['R'], D_real['dI'], D_real['dR']

    P_sim = 1 - jnp.exp(-beta * (I_real / population))
    part1 = delta_I_real * jnp.log(P_sim)
    part2 = -beta * I_real / population * (S_real - delta_I_real)
    # The scale is used to make MCMC stable
    return (log_prior_beta + (part1 + part2).sum()) / log_posterior_scale


def SIR(rng_key):
    Nx_list = [10]
    Ny_list = [5, 10, 20, 30, 40]

    population = float(1e4)
    beta_real, gamma_real = 0.5, 0.05
    beta_lab, gamma_lab = 0.45, 0.05
    rate = 10.0
    scale = 1. / rate
    T = 150
    target_date = 20

    D_real, D_real_target = generate_data(beta_real, gamma_real, T, population, target_date, rng_key)
    D_real_target = SIR_utils.convert_dict_to_jnp(D_real_target)
    N_MCMC = 1000
    init_params = beta_lab
    # This one is heuristic
    log_posterior_scale = population / 100
    log_posterior_fn = partial(log_posterior, gamma=gamma_lab, D_real=D_real_target,
                               population=population, beta_lab=beta_lab, T=T,
                               log_posterior_scale=log_posterior_scale, rng_key=rng_key)
    grad_log_posterior_fn = jax.grad(log_posterior_fn)
    samples_post = MCMC(rng_key, beta_lab, N_MCMC, init_params, log_posterior_fn)
    samples_post = jnp.unique(samples_post, axis=0)
    rng_key, _ = jax.random.split(rng_key)
    samples_post = jax.random.permutation(rng_key, samples_post)

    # Debug : Large sample Monte Carlo
    beta_array_large_sample = jnp.zeros([N_MCMC, 1])
    f_beta_array_large_sample = jnp.zeros([N_MCMC, 1])
    print(f"Sampling {N_MCMC} with MCMC to estimate the true value")
    for i in tqdm(range(N_MCMC)):
        beta = samples_post[i]
        D, _ = generate_data(beta, gamma_real, T, population, target_date, rng_key)
        f_beta = D['I'][target_date + 1]
        beta_array_large_sample = beta_array_large_sample.at[i, :].set(beta)
        f_beta_array_large_sample = f_beta_array_large_sample.at[i, :].set(f_beta)
    MC_large_sample = f_beta_array_large_sample.mean()

    for Ny in Ny_list:
        beta_array = jnp.zeros([Ny, 1])
        f_beta_array = jnp.zeros([Ny])
        d_log_beta_array = jnp.zeros([Ny, 1])
        for i in range(Ny):
            beta = samples_post[i]
            D, _ = generate_data(beta, gamma_real, T, population, target_date, rng_key)
            f_beta = D['I'][target_date + 1]
            d_log_beta = grad_log_posterior_fn(beta) * log_posterior_scale
            beta_array = beta_array.at[i, :].set(beta)
            f_beta_array = f_beta_array.at[i].set(f_beta)
            d_log_beta_array = d_log_beta_array.at[i, :].set(d_log_beta)

        MC = Monte_Carlo(f_beta_array)
        f_beta_array_mean, f_beta_array_std, f_beta_array_standardized = SIR_utils.scale(f_beta_array)
        BMC_mean, BMC_std = Bayesian_Monte_Carlo(rng_key, beta_array, f_beta_array_standardized, d_log_beta_array, stein_Matern)
        BMC_mean = BMC_mean * f_beta_array_std + f_beta_array_mean
        BMC_std = BMC_std * f_beta_array_std
        # Debug
        print('True value (MC with large samples)', MC_large_sample)
        print(f'MC with {Ny} number of Y', MC)
        print(f'BMC with {Ny} number of Y', BMC_mean)
        print(f"=================")
        pause = True
    return


def main():
    seed = int(time.time())
    rng_key = jax.random.PRNGKey(seed)
    SIR(rng_key)
    return


if __name__ == '__main__':
    main()
