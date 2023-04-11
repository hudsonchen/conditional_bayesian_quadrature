import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import jax
import numpy as np
import pandas as pd
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm
tfd = tfp.distributions

eps = 1e-6


def convert_dict_to_jnp(D):
    D_copy = {}
    for k in D.keys():
        D_copy[k] = jnp.array(D[k])
    return D_copy


def non_zero_ind(A):
    ind = jnp.abs(A) > 2.0
    return ind


def scale(A):
    m = A.mean()
    return m, A / m


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    # mean = 0.
    # std = 1.
    standardized = (Z - mean) / std
    return standardized, mean, std


@jax.jit
def time_step(beta, gamma, population, St, It, Rt, dt, rng_key):
    dS = -beta * St * It * dt / population
    dI = (beta * St * It / population - gamma * It) * dt
    dR = gamma * It * dt
    St = St + dS
    It = It + dI
    Rt = Rt + dR
    return St, It, Rt


def generate_data(beta, gamma, T, dt, population, rng_key):
    """
    :param beta: float, infection rate
    :param gamma: float, recovery rate
    :param rng_key:
    :param T: Time length
    :return: array T*3, the first is number of Susceptible,
    the second is Infected, the third is Recoverdd
    """
    It, Rt = 50., 0.
    St = population - It - Rt

    S_list = []
    I_list = []
    R_list = []
    iter_ = int(T / dt)

    for i in range(iter_):
        S_list.append(St)
        I_list.append(It)
        R_list.append(Rt)
        St, It, Rt = time_step(beta, gamma, population, St, It, Rt, dt, rng_key)

    S_array = np.array(S_list)
    I_array = np.array(I_list)
    R_array = np.array(R_list)

    # # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(np.arange(iter_), S_array, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(np.arange(iter_), I_array, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(np.arange(iter_), R_array, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.title(f'Infection rate is {beta}')
    plt.show()

    D_real = {'S': S_array, 'I': I_array, 'R': R_array}
    pause = True
    return D_real


def ground_truth_peak_infected_number(beta_all, gamma, D_real, T, population, MCMC_fn, N_MCMC, log_posterior,
                                      rate, rng_key):
    peak_infected_number = jnp.zeros_like(beta_all)
    for i, beta in enumerate(tqdm(beta_all)):
        rng_key, _ = jax.random.split(rng_key)
        log_posterior_fn = partial(log_posterior, beta_mean=0., beta_std=1.0, gamma=gamma, D_real=D_real,
                                   population=population, beta_0=beta, rate=rate, rng_key=rng_key)

        samples = MCMC_fn(rng_key, beta, N_MCMC, beta, log_posterior_fn, rate)
        samples = jnp.unique(samples, axis=0)
        rng_key, _ = jax.random.split(rng_key)
        samples = jax.random.permutation(rng_key, samples)

        dummy = 0
        for s in samples:
            rng_key, _ = jax.random.split(rng_key)
            D = generate_data(s, gamma, T, population, rng_key)
            dummy += D['dI'].max()
        peak_infected_number = peak_infected_number.at[i].set(dummy / len(samples))
    return peak_infected_number


def ground_truth_peak_infected_time(beta_all, gamma, D_real, T, population, MCMC_fn, N_MCMC, log_posterior,
                                      rate, rng_key):
    peak_infected_time = jnp.zeros_like(beta_all)
    for i, beta in enumerate(tqdm(beta_all)):
        rng_key, _ = jax.random.split(rng_key)
        log_posterior_fn = partial(log_posterior, beta_mean=0., beta_std=1.0, gamma=gamma, D_real=D_real,
                                   population=population, beta_0=beta, rate=rate, rng_key=rng_key)

        samples = MCMC_fn(rng_key, beta, N_MCMC, beta, log_posterior_fn, rate)
        samples = jnp.unique(samples, axis=0)
        rng_key, _ = jax.random.split(rng_key)
        samples = jax.random.permutation(rng_key, samples)

        dummy = 0
        for s in samples:
            rng_key, _ = jax.random.split(rng_key)
            D = generate_data(s, gamma, T, population, rng_key)
            dummy += D['dI'].argmax()
        peak_infected_time = peak_infected_time.at[i].set(dummy / len(samples))
    return peak_infected_time
