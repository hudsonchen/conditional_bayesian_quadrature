import jax
import jax.numpy as jnp
import os
import pwd
import jax.scipy
import jax.scipy.stats
from functools import partial
import matplotlib.pyplot as plt
import MCMC
from tqdm import tqdm
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from kernels import *
import optax
from utils import finance_utils
import time
from jax.config import config
config.update("jax_enable_x64", True)


if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
else:
    pass

eps = 1e-6


def generate_date(rng_key, num):
    rng_key, _ = jax.random.split(rng_key)
    x_1 = jax.random.uniform(rng_key, shape=(num, 1), minval=-1.0, maxval=1.0)
    rng_key, _ = jax.random.split(rng_key)
    x_2 = jax.random.uniform(rng_key, shape=(num, 1), minval=-1.0, maxval=1.0)
    p = 1. / (1. + jnp.exp(-(x_1 + x_2)))
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.bernoulli(rng_key, p)
    jnp.save(f'./data/sensitivity/data_y', Y)
    X = jnp.concatenate((x_1, x_2), axis=1)
    jnp.save(f'./data/sensitivity/data_x', X)
    return


def log_posterior(beta, x, y, prior_cov):
    """
    :param prior_cov: 3*1 array
    :param beta: 3*1 array
    :param x: N*2 array
    :param y: N*1 array
    :return:
    """
    prior_cov = jnp.diag(prior_cov.squeeze())
    log_prior_beta = jax.scipy.stats.multivariate_normal.logpdf(beta.squeeze(), mean=jnp.zeros([3]), cov=prior_cov).sum()
    x_with_one = jnp.hstack([x, jnp.ones([x.shape[0], 1])])
    p = jax.nn.sigmoid(x_with_one @ beta)
    log_bern_llk = (y * jnp.log(p + eps) + (1 - y) * jnp.log(1 - p + eps)).sum()
    return (log_bern_llk + log_prior_beta).squeeze()


def MCMC(rng_key, nsamples, init_params, log_prob):
    rng_key, _ = jax.random.split(rng_key)

    @jax.jit
    def run_chain(rng_key, state):
        num_burnin_steps = int(100)
        # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=3,
        #         step_size=1.0),
        #         num_adaptation_steps=int(num_burnin_steps * 0.8))

        kernel = tfp.mcmc.NoUTurnSampler(log_prob, 1e-1)
        return tfp.mcmc.sample_chain(num_results=nsamples,
                                     num_burnin_steps=num_burnin_steps,
                                     current_state=state,
                                     kernel=kernel,
                                     trace_fn=None,
                                     seed=rng_key)
    states = run_chain(rng_key, init_params)
    # Debug code

    fig = plt.figure(figsize=(15, 6))
    ax_0, ax_1, ax_2 = fig.subplots(1, 3)

    x = jnp.linspace(-3 * 10, 3 * 10, 100)
    beta_0_post = states[:, 0, :]
    ax_0.plot(x, jax.scipy.stats.norm.pdf(x, 0, 10), color='black', linewidth=5)
    ax_0.hist(np.array(beta_0_post), bins=10, alpha=0.8, density=True)

    x = jnp.linspace(-3 * 2.5, 3 * 2.5, 100)
    beta_1_post = states[:, 1, :]
    ax_1.plot(x, jax.scipy.stats.norm.pdf(x, 0, 2.5), color='black', linewidth=5)
    ax_1.hist(np.array(beta_1_post), bins=10, alpha=0.8, density=True)

    x = jnp.linspace(-3 * 2.5, 3 * 2.5, 100)
    beta_2_post = states[:, 2, :]
    ax_2.plot(x, jax.scipy.stats.norm.pdf(x, 0, 2.5), color='black', linewidth=5)
    ax_2.hist(np.array(beta_2_post), bins=10, alpha=0.8, density=True)
    plt.show()
    return states


def g(y):
    """
    :param y: y is a N*3*1 array
    """
    return (y ** 2).sum(1).squeeze(axis=-1)


def Monte_Carlo(gy):
    return gy.mean(0)


def stein_Matern(x, y, l, d_log_px, d_log_py):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :param d_log_px: N*D
    :param d_log_py: M*D
    :return: N*M
    """
    N, D = x.shape
    M = y.shape[0]

    batch_kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=l)
    grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=0)
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=0)
    grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=1)
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=0)

    grad_xy_K_fn = jax.jacfwd(jax.jacrev(batch_kernel.apply, argnums=1), argnums=0)

    def diag_sum_grad_xy_K_fn(x, y):
        return jnp.diag(grad_xy_K_fn(x, y)).sum()

    vec_grad_xy_K_fn = jax.vmap(diag_sum_grad_xy_K_fn, in_axes=(0, 0), out_axes=0)

    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)

    K = batch_kernel.matrix(x, y)
    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    dxdy_K = vec_grad_xy_K_fn(x_dummy, y_dummy).reshape(N, M)

    part1 = d_log_px @ d_log_py.T * K
    part2 = (d_log_py[None, :] * dx_K).sum(-1)
    part3 = (d_log_px[:, None, :] * dy_K).sum(-1)
    part4 = dxdy_K

    return part1 + part2 + part3 + part4


def Bayesian_Monte_Carlo(rng_key, y, gy, d_log_py):
    """
    :param rng_key:
    :param y: N * D * 1
    :param gy: N
    :param d_log_py: N * D * 1
    :return:
    """
    y = y[:, :, 0]
    d_log_py = d_log_py[:, :, 0]
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    log_l_init = log_l = jnp.log(0.5)
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l = jnp.exp(log_l)
        n = y.shape[0]
        K = A * stein_Matern(y, y, l, d_log_py, d_log_py) + c
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps))
        return nll

    @jax.jit
    def step(log_l, c, A, opt_state, rng_key):
        nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1, 2))(log_l, c, A)
        updates, opt_state = optimizer.update(grads, opt_state, (log_l, c, A))
        log_l, c, A = optax.apply_updates((log_l, c, A), updates)
        return log_l, c, A, opt_state, nllk_value

    # Debug code
    log_l_debug_list = []
    c_debug_list = []
    A_debug_list = []
    nll_debug_list = []
    for _ in range(3000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # Debug code
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

    l = jnp.exp(log_l)
    final_K = A * stein_Matern(y, y, l, d_log_py, d_log_py) + c
    final_K_inv = jnp.linalg.inv(final_K + eps * jnp.eye(n))
    BMC_mean = c * (final_K_inv @ gy).sum()
    BMC_std = jnp.sqrt(c - final_K_inv.sum() * c * c)

    if jnp.isnan(BMC_std):
        BMC_std = 0.3
    return BMC_mean, BMC_std


def main():
    seed = int(time.time())
    rng_key = jax.random.PRNGKey(seed)
    # generate_date(rng_key, 30)
    X = jnp.load(f'./data/sensitivity/data_x.npy')
    Y = jnp.load(f'./data/sensitivity/data_y.npy')

    N_alpha_list = [3]
    N_beta_list = [30]

    alpha_test = jax.random.uniform(rng_key, shape=(3, 1), minval=-1.0, maxval=1.0)

    for n_alpha in N_alpha_list:
        rng_key, _ = jax.random.split(rng_key)
        alpha = jax.random.uniform(rng_key, shape=(3, 1), minval=-1.0, maxval=1.0)
        cov = jnp.array([[10, 2.5, 2.5]]).T + alpha
        log_prob = partial(log_posterior, x=X, y=Y, prior_cov=cov)
        grad_log_prob = jax.grad(log_prob, argnums=0)

        init_params = jnp.array([[0., 0., 0.]]).T
        states_true = MCMC(rng_key, 4000, init_params, log_prob)
        states_true = jnp.unique(states_true, axis=0)
        rng_key, _ = jax.random.split(rng_key)
        states_true = jax.random.permutation(rng_key, states_true)
        print(g(states_true).mean())

        for n_beta in N_beta_list:
            states = states_true[:n_beta, :]
            g_states = g(states)
            d_log_pstates = grad_log_prob(states)

            # True value
            MC = Monte_Carlo(g_states)
            print(MC)
            states_standardized, states_mean, state_std = finance_utils.standardize(states)
            BMC_mean, BMC_std = Bayesian_Monte_Carlo(rng_key, states, g_states, d_log_pstates)
    pause = True
    return


if __name__ == '__main__':
    main()
