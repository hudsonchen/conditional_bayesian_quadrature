import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import jax
import pickle
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
    # m = 10000
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

    iter_ = int(T / dt)

    S_array = jnp.zeros([iter_])
    I_array = jnp.zeros([iter_])
    R_array = jnp.zeros([iter_])

    for i in range(iter_):
        S_array = S_array.at[i].set(St)
        I_array = I_array.at[i].set(It)
        R_array = R_array.at[i].set(Rt)
        St, It, Rt = time_step(beta, gamma, population, St, It, Rt, dt, rng_key)

    # # Plot the data on three separate curves for S(t), I(t) and R(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    # ax.plot(np.arange(iter_), S_array, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(np.arange(iter_), I_array, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.plot(np.arange(iter_), R_array, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    # ax.set_xlabel('Time /days')
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # plt.title(f'Infection rate is {beta}')
    # plt.show()

    D = {'S': S_array, 'I': I_array, 'R': R_array}
    pause = True
    return D['I']


def save(args, Nx, Ny, beta_0_test, BMC_mean_array, BMC_mean, BMC_std, KMS_mean, LSMC_mean, IS_mean,
         ground_truth_array, beta_0_array, BMC_time, KMS_time, LSMC_time, IS_time):
    # jnp.save(f"{args.save_path}/BMC_mean.npy", BMC_mean.squeeze())
    # jnp.save(f"{args.save_path}/BMC_std.npy", BMC_std.squeeze())
    # jnp.save(f"{args.save_path}/KMS_mean.npy", KMS_mean.squeeze())
    # jnp.save(f"{args.save_path}/LSMC_mean.npy", LSMC_mean.squeeze())

    time_dict = {'BMC': BMC_time, 'IS': IS_time, 'LSMC': LSMC_time, 'KMS': KMS_time}
    with open(f"{args.save_path}/time_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(time_dict, f)

    mse_dict = {}
    mse_dict['BMC'] = ((ground_truth_array - BMC_mean) ** 2).mean()
    mse_dict['IS'] = ((ground_truth_array - IS_mean) ** 2).mean()
    mse_dict['LSMC'] = ((ground_truth_array - LSMC_mean) ** 2).mean()
    mse_dict['KMS'] = ((ground_truth_array - KMS_mean) ** 2).mean()
    with open(f"{args.save_path}/mse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(mse_dict, f)

    # ========== Debug code ==========
    plt.figure()
    plt.plot(beta_0_test, BMC_mean, color='blue', label='BMC')
    plt.plot(beta_0_test, KMS_mean, color='red', label='KMS')
    plt.plot(beta_0_test, LSMC_mean, color='green', label='LSMC')
    plt.plot(beta_0_test, IS_mean, color='orange', label='IS')
    plt.plot(beta_0_test, ground_truth_array, color='black', label='True')
    plt.scatter(beta_0_array, BMC_mean_array, color='orange')
    plt.fill_between(beta_0_test, BMC_mean - BMC_std, BMC_mean + BMC_std, alpha=0.2, color='blue')
    plt.legend()
    plt.title(f"Nx={Nx}, Ny={Ny}")
    plt.savefig(f"{args.save_path}/figures/SIR_X_{Nx}_y_{Ny}.pdf")
    plt.show()
    plt.close()
    pause = True
    # ========== Debug code ==========
    return


def save_large(args, Nx, Ny, KMS_mean, LSMC_mean, IS_mean, ground_truth_array, KMS_time, LSMC_time, IS_time):
    time_dict = {'BMS': None, 'IS': IS_time, 'LSMC': LSMC_time, 'KMS': KMS_time}
    with open(f"{args.save_path}/time_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(time_dict, f)

    mse_dict = {}
    mse_dict['IS'] = ((ground_truth_array - IS_mean) ** 2).mean()
    mse_dict['LSMC'] = ((ground_truth_array - LSMC_mean) ** 2).mean()
    mse_dict['KMS'] = ((ground_truth_array - KMS_mean) ** 2).mean()
    mse_dict['BMS'] = None
    with open(f"{args.save_path}/mse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(mse_dict, f)
    return


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
