import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy import integrate
import time
import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import tqdm
from finance_baselines import *
from kernels import *
from utils import finance_utils
import os
os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")


plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def Geometric_Brownian(n, dt, rng_key, sigma=0.3, S0=1):
    rng_key, _ = jax.random.split(rng_key)
    dWt = jax.random.normal(rng_key, shape=(int(n),)) * jnp.sqrt(dt)
    dlnSt = sigma * dWt - 0.5 * (sigma ** 2) * dt
    St = jnp.exp(jnp.cumsum(dlnSt) + jnp.log(S0))
    return St


def callBS(t, s, K, sigma):
    part_one = jnp.log(s / K) / (sigma * jnp.sqrt(t)) + 0.5 * sigma * jnp.sqrt(t)
    part_two = jnp.log(s / K) / (sigma * jnp.sqrt(t)) - 0.5 * sigma * jnp.sqrt(t)
    return s * norm.cdf(part_one) - K * norm.cdf(part_two)


def BSM_butterfly_analytic():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50

    # St = Geometric_Brownian(100, t / 100, sigma=sigma, S0=S0)[-1]

    epsilon = jax.random.normal(rng_key)
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    psiST_St_1 = callBS(T - t, St, K1, sigma) + callBS(T - t, St, K2, sigma) \
                 - 2 * callBS(T - t, St, (K1 + K2) / 2, sigma)
    psiST_St_2 = callBS(T - t, (1 + s) * St, K1, sigma) + callBS(T - t, (1 + s) * St, K2, sigma) \
                 - 2 * callBS(T - t, (1 + s) * St, (K1 + K2) / 2, sigma)
    L_inner = psiST_St_1 - psiST_St_2
    # print(max(L_inner, 0))
    print(L_inner)

    # Verifies the result from BSM agree with standard Monte Carlo
    # iter_num = 10000
    # L_MC = 0
    # for i in range(iter_num):
    #     epsilon = np.random.normal(0, 1)
    #     ST = St * np.exp(sigma * np.sqrt((T-t)) * epsilon - 0.5 * (sigma ** 2) * (T-t))
    #     psi_ST_1 = max(ST - K1, 0) + max(ST - K2, 0) - 2 * max(ST - (K1+K2) / 2, 0)
    #     psi_ST_2 = max((1+s) * ST - K1, 0) + max((1+s) * ST - K2, 0) - 2 * max((1+s) * ST - (K1+K2) / 2, 0)
    #     L_MC += (psi_ST_1 - psi_ST_2)
    # print(L_MC / iter_num)
    return


@jax.jit
def dx_log_px(x, sigma, T, t, scale, St):
    # dx log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - y)
    x *= scale
    part1 = (jnp.log(x) + sigma ** 2 * (T - t) / 2 - jnp.log(St)) / x / (sigma ** 2 * (T - t))
    return (-1. / x - part1) * scale


@jax.jit
def stein_Matern(x, y, l, sigma, T, t, scale, St):
    d_log_px = dx_log_px(x, sigma, T, t, scale, St)
    d_log_py = dx_log_px(y, sigma, T, t, scale, St)

    K = my_Matern(x, y, l)
    dx_K = dx_Matern(x, y, l)
    dy_K = dy_Matern(x, y, l)
    dxdy_K = dxdy_Matern(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


@jax.jit
def stein_Laplace(x, y, l, sigma, T, t, scale, St):
    d_log_px = dx_log_px(x, sigma, T, t, scale, St)
    d_log_py = dx_log_px(y, sigma, T, t, scale, St)

    K = my_Laplace(x, y, l)
    dx_K = dx_Laplace(x, y, l)
    dy_K = dy_Laplace(x, y, l)
    dxdy_K = dxdy_Laplace(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


def train(y, y_scale, gy, rng_key, sigma, T, t, St):
    """
    :param y:
    :param gy:
    :return: the hyperparamter for stein kernel
    """
    rng_key, _ = jax.random.split(rng_key)
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    log_l_init = log_l = jnp.log(0.3)
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l = jnp.exp(log_l)
        n = y.shape[0]
        K = A_init * stein_Laplace(y, y, l, sigma, T, t, y_scale, St) + c + eps * jnp.eye(n)
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / n
        return nll[0][0]

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
    #     # Debug code
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

    # l = jnp.exp(log_l)
    # y_debug = jnp.linspace(20, 160, 100)[:, None] / y_scale
    # K_train_train = stein_Laplace(y, y, l, sigma, T=2, t=1, scale=y_scale, St=St) + eps * jnp.eye(n) + c
    # K_train_train_inv = jnp.linalg.inv(K_train_train)
    # K_test_train = stein_Laplace(y_debug, y, l, sigma, T=2, t=1, scale=y_scale, St=St) + c
    # gy_debug = K_test_train @ K_train_train_inv @ gy
    # plt.figure()
    # plt.scatter(y * y_scale, gy)
    # plt.plot(y_debug * y_scale, gy_debug)
    # plt.show()
    pause = True
    return jnp.exp(log_l), c, A


def cbq(X, Y, gY, x_prime, rng_key, sigma, kernel_y, compute_phi='empirical'):
    """
    :param X: X is of size Nx
    :param Y: Y is of size Nx * Ny
    :param gY: gY is g(Y)
    :param x_prime: is the target conditioning value of x, should be of shape [1, 1]
    :param compute_phi: choose the mode to compute kernel mean embedding phi.
    :return: return the expectation E[g(Y)|X=x_prime]
    """

    Nx = X.shape[0]
    Ny = Y.shape[1]
    eps = 1e-6

    if kernel_y == 'rbf':
        Ky = my_RBF
        ly = 0.5
    elif kernel_y == 'matern':
        Ky = my_Matern
        ly = 0.5
    elif kernel_y == 'laplace':
        Ky = my_Laplace
        ly = 0.5
    elif kernel_y == 'stein_matern':
        Ky = stein_Matern
    elif kernel_y == 'stein_laplace':
        Ky = stein_Laplace
    else:
        raise NotImplementedError

    mu_list = []
    std_list = []
    for i in range(Nx):
        x = X[i]
        Yi = Y[i, :][:, None]
        Yi_standardized, Yi_scale = finance_utils.scale(Yi)
        gYi = gY[i, :][:, None]

        ly, c, A = train(Yi_standardized, Yi_scale, gYi, rng_key, sigma, T=2, t=1, St=x)
        # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'

        # Ky_inv = jnp.linalg.inv(Ky(Yi_standardized, Yi_standardized, ly) + eps * jnp.eye(Ny))
        K = A * Ky(Yi_standardized, Yi_standardized, ly, sigma, T=2, t=1, scale=Yi_scale, St=x)
        K_inv = jnp.linalg.inv(A * Ky(Yi_standardized, Yi_standardized, ly, sigma, T=2, t=1, scale=Yi_scale, St=x) +
                              c + eps * jnp.eye(Ny))
        mu_standardized = c * (K_inv @ gYi).sum()
        std_standardized = jnp.sqrt(c - K_inv.sum() * c ** 2)

        mu_list.append(mu_standardized.squeeze())
        std_list.append(std_standardized.squeeze())

        # Large sample mu
        # print(price(X[i], 10000, rng_key)[1].mean())
    Sigma = jnp.array(std_list)
    Mu = jnp.array(mu_list)
    return Mu, Sigma


@partial(jax.jit, static_argnums=(4, 5))
def GP(psi_y_x_mean, psi_y_x_std, X, x_prime, Kx, one_d_Kx, lx):
    Nx = psi_y_x_mean.shape[0]

    Mu_standardized, Mu_mean, Mu_std = finance_utils.standardize(psi_y_x_mean)
    Sigma_standardized = psi_y_x_std / Mu_std
    X_standardized, X_mean, X_std = finance_utils.standardize(X)
    x_prime_standardized = (x_prime - X_mean) / X_std
    noise = 0.1

    K_train_train = Kx(X_standardized, X_standardized, lx) + jnp.diag(Sigma_standardized) + noise * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = one_d_Kx(x_prime_standardized, X_standardized, lx)
    K_test_test = one_d_Kx(x_prime_standardized, x_prime_standardized, lx) + noise
    mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
    var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x_prime = jnp.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
    std_y_x_prime_original = std_y_x_prime * Mu_std

    # x_debug = jnp.linspace(20, 120, 100)[:, None]
    # x_debug_standardized = (x_debug - X_mean) / X_std
    # K_train_debug = Kx(X_standardized, x_debug_standardized, lx)
    # mu_y_x_debug = K_train_debug.T @ K_train_train_inv @ Mu_standardized
    # var_y_x_debug = Kx(x_debug_standardized, x_debug_standardized, lx) - K_train_debug.T @ K_train_train_inv @ K_train_debug
    # std_y_x_debug = jnp.sqrt(jnp.diag(var_y_x_debug))
    # mu_y_x_debug_original = mu_y_x_debug * Mu_std + Mu_mean
    # std_y_x_debug_original = std_y_x_debug * Mu_std
    #
    # true_X = jnp.load('./data/finance_X.npy')
    # true_EgY_X = jnp.load('./data/finance_EgY_X.npy')

    # plt.figure()
    # plt.plot(x_debug.squeeze(), mu_y_x_debug_original.squeeze(), label='predict')
    # plt.plot(true_X, true_EgY_X, label='true')
    # plt.scatter(X.squeeze(), psi_y_x_mean.squeeze())
    # plt.fill_between(x_debug.squeeze(), mu_y_x_debug_original.squeeze() - std_y_x_debug_original,
    #                  mu_y_x_debug_original.squeeze() + std_y_x_debug_original, alpha=0.5)
    # plt.plot()
    # plt.legend()
    # plt.savefig(f"./results/GP_finance_{Nx}.pdf")
    # plt.show()
    return mu_y_x_prime_original, std_y_x_prime_original


def price(St, N, rng_key, K1=50, K2=150, s=-0.2, sigma=0.3, T=2, t=1, visualize=False):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)

    if visualize:
        loss_list = []
        ST_list = []
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs = axs.flatten()

        St_dummy = St[0]
        log_ST_min = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (-3) - 0.5 * (sigma ** 2) * (T - t)
        log_ST_max = jnp.log(St_dummy) + sigma * jnp.sqrt((T - t)) * (+3) - 0.5 * (sigma ** 2) * (T - t)
        ST_min = jnp.exp(log_ST_min)
        ST_max = jnp.exp(log_ST_max)
        ST_samples = jnp.linspace(ST_min, ST_max, 100)
        normal_samples = (jnp.log(ST_samples) - jnp.log(St_dummy) + 0.5 * (sigma ** 2) * (T - t)) / sigma
        density = norm.pdf(normal_samples)
        axs[0].plot(ST_samples, density)
        axs[0].set_title(r"The pdf for $p(S_T|S_t)$")
        axs[0].set_xlabel(r"$S_T$")

        for _ in range(1000):
            rng_key, _ = jax.random.split(rng_key)
            epsilon = jax.random.normal(rng_key, shape=(1, 1))
            ST = St_dummy * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
            psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
            psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * np.maximum(
                (1 + s) * ST - (K1 + K2) / 2, 0)
            loss_list.append(psi_ST_1 - psi_ST_2)
            ST_list.append(ST)

        ST_dummy = jnp.array(ST_list).squeeze()
        loss_dummy = jnp.array(loss_list).squeeze()
        axs[1].scatter(ST_dummy, loss_dummy)
        axs[1].set_title(r"$\psi(S_T) - \psi((1+s)S_T)$")
        axs[1].set_xlabel(r"$S_T$")
        plt.suptitle(rf"$S_t$ is {St_dummy[0]}")
        plt.savefig(f"./results/finance_dummy.pdf")
        # plt.show()
    return ST, psi_ST_1 - psi_ST_2


def save_true_value():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50
    epsilon = jax.random.normal(rng_key, shape=(1000, 1))
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    _, loss = price(St, 100000, rng_key, K1=K1, K2=K2, s=s, sigma=sigma, T=T, t=t)
    St = St.squeeze()
    ind = jnp.argsort(St)
    value = loss.mean(1)
    jnp.save('./data/finance_X.npy', St[ind])
    jnp.save('./data/finance_EgY_X.npy', value[ind])
    plt.figure()
    plt.plot(St[ind], value[ind])
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\mathbb{E}[g(Y) \mid X]$")
    plt.title("True value for finance experiment")
    plt.savefig("./data/true_distribution.pdf")
    plt.show()
    return


def cbq_option_pricing(visualize=False):
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)

    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50
    Nx_array = jnp.array([5, 10, 20])
    # Ny_array = jnp.arange(2, 100, 2)
    Ny_array = jnp.array([3, 5, 10, 30, 50, 70, 90, 110])
    cbq_mean_dict = {}
    cbq_std_dict = {}
    poly_mean_dict = {}
    poly_std_dict = {}
    MC_list = []

    St_prime = jnp.array([[70.0]])
    if visualize:
        rng_key, _ = jax.random.split(rng_key)
        price(St_prime, N=1, rng_key=rng_key, visualize=True)

    # True value with standard MC
    for _ in range(1):
        rng_key, _ = jax.random.split(rng_key)
        true_value = price(St_prime, 1000000, rng_key)[1].mean()
        print('True Value is:', true_value)

    kernel_x = 'rbf'
    if kernel_x == 'rbf':  # This is the best kernel for x
        Kx = my_RBF
        one_d_Kx = one_d_my_RBF
        lx = 1.0
    elif kernel_x == 'matern':
        Kx = my_Matern
        one_d_Kx = one_d_my_Matern
        lx = 0.5
    else:
        raise NotImplementedError

    for Nx in Nx_array:
        cbq_mean_array = jnp.array([])
        cbq_std_array = jnp.array([])
        poly_mean_array = jnp.array([])
        poly_std_array = jnp.array([])
        for Ny in tqdm(Ny_array):
            rng_key, _ = jax.random.split(rng_key)
            epsilon = jax.random.normal(rng_key, shape=(Nx, 1))
            St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
            ST, loss = price(St, Ny, rng_key, K1=K1, K2=K2, s=s, sigma=sigma, T=T, t=t)

            # This is use standard GP to fit loss at a given St, mainly used for choosing kernel on Y.
            # GP(St, ST, loss, price, St_prime, kernel_x='rbf', kernel_y='exponential')

            psi_x_mean, psi_x_std = cbq(St, ST, loss, St_prime, rng_key, sigma=sigma,
                                                      kernel_y='stein_matern')
            psi_x_std = np.nan_to_num(psi_x_std, nan=1.0)
            mu_y_x_prime_cbq, std_y_x_prime_cbq = GP(psi_x_mean, psi_x_std, St, St_prime, Kx, one_d_Kx, lx)

            mu_y_x_prime_poly, std_y_x_prime_poly = polynommial(St, ST, loss, St_prime, sigma=sigma)
            cbq_mean_array = jnp.append(cbq_mean_array, mu_y_x_prime_cbq)
            cbq_std_array = jnp.append(cbq_std_array, std_y_x_prime_cbq)
            poly_mean_array = jnp.append(poly_mean_array, mu_y_x_prime_poly)
            poly_std_array = jnp.append(poly_std_array, std_y_x_prime_poly)
        cbq_mean_dict[f"{Nx}"] = cbq_mean_array
        cbq_std_dict[f"{Nx}"] = cbq_std_array
        poly_mean_dict[f"{Nx}"] = poly_mean_array
        poly_std_dict[f"{Nx}"] = poly_std_array

    for Ny in Ny_array:
        rng_key, _ = jax.random.split(rng_key)
        MC_list.append(price(St_prime, Ny, rng_key)[1].mean())

    fig, axs = plt.subplots(len(Nx_array), 1, figsize=(10, len(Nx_array) * 3))
    for i, ax in enumerate(axs):
        Nx = Nx_array[i]
        # axs[i].set_ylim(1, 7)
        axs[i].axhline(y=true_value, linestyle='--', color='black', label='true value')
        axs[i].plot(Ny_array, MC_list, color='b', label='MC')
        axs[i].plot(Ny_array, cbq_mean_dict[f"{Nx}"], color='r', label=f'CBQ Nx = {Nx}')
        axs[i].plot(Ny_array, poly_mean_dict[f"{Nx}"], color='orange', label=f'Polynomial Nx = {Nx}')
        axs[i].fill_between(Ny_array, cbq_mean_dict[f"{Nx}"] - 2 * cbq_std_dict[f"{Nx}"],
                            cbq_mean_dict[f"{Nx}"] + 2 * cbq_std_dict[f"{Nx}"], color='r', alpha=0.5)
        axs[i].legend()
        # axs[i].set_xscale('log')
    plt.tight_layout()
    plt.savefig("./results/CBQ_results_finance.pdf")
    # plt.show()
    return


def main():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)

    visualize_brownian = False
    debug_BSM = False
    debug_cbq = True
    if visualize_brownian:
        n = 100.
        T = 1.
        dt = T / n
        plt.figure()
        for i in range(10):
            St = Geometric_Brownian(n, dt, rng_key)
            plt.plot(St)
        plt.show()
    elif debug_BSM:
        BSM_butterfly_analytic()
    elif debug_cbq:
        cbq_option_pricing(visualize=False)
    else:
        pass
    return


if __name__ == '__main__':
    main()
    # save_true_value()
#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
