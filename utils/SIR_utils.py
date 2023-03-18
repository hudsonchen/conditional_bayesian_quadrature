import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import jax
import numpy as np
import pandas as pd
from tensorflow_probability.substrates import jax as tfp
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
    m = 0.
    s = jnp.log10(A.max())
    std = 10 ** (int(s))
    return m, std, (A - m) / std


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / (std + eps)
    return standardized, mean, std


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
    D_real_remove_zero = {'S': S_array[10:target_date], 'I': I_array[10:target_date],
                          'R': R_array[10:target_date], 'dI': delta_It_array[10:target_date],
                          'dR': delta_Rt_array[:target_date]}
    return D_real, D_real_remove_zero


def ground_truth_peak_infected_number(beta_lab_all, gamma, T, population, target_date, rng_key):
    peak_infected_number = jnp.zeros_like(beta_lab_all)
    for i, beta in enumerate(beta_lab_all):
        D, _ = generate_data(beta, gamma, T, population, target_date, rng_key)
        peak_infected_number = peak_infected_number.at[i].set(D['I'].max())
    return peak_infected_number


def ground_truth_peak_infected_time(beta_lab_all, gamma, T, population, target_date, rng_key):
    peak_infected_time = jnp.zeros_like(beta_lab_all)
    for i, beta in enumerate(beta_lab_all):
        D, _ = generate_data(beta, gamma, T, population, target_date, rng_key)
        peak_infected_time = peak_infected_time.at[i].set(D['I'].argmax())
    return peak_infected_time
