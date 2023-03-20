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
import pwd
import argparse
import pickle
from jax.config import config

config.update("jax_enable_x64", True)

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
    print(os.getcwd())
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
    print(os.getcwd())
else:
    pass

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for finance data')

    # Data settings
    parser.add_argument('--kernel_x', type=str)
    parser.add_argument('--kernel_y', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    return args


@jax.jit
def grad_y_log_py_x(y, x, y_scale, sigma, T, t):
    # dx log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - y)
    y *= y_scale
    part1 = (jnp.log(y) + sigma ** 2 * (T - t) / 2 - jnp.log(x)) / y / (sigma ** 2 * (T - t))
    return (-1. / y - part1) * y_scale


@jax.jit
def py_x_fn(y, x, y_scale, sigma, T, t):
    """
    :param y: Ny * 1
    :param x: scalar
    :param y_scale: scalar
    :return: scalar
    """
    # dx log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    y_tilde = y * y_scale
    z = jnp.log(y_tilde / x)
    n = (z + sigma ** 2 * (T - t) / 2) / sigma / jnp.sqrt(T - t)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T - t))
    p_y_tilde = p_z / y_tilde
    p_y = p_y_tilde / y_scale
    return p_y


@jax.jit
def log_py_x_fn(y, x, y_scale, sigma, T, t):
    # dx log p(x) for log normal distribution with mu=-\sigma^2 / 2 * (T - t) and sigma = \sigma^2 (T - t)
    y_tilde = y * y_scale
    z = jnp.log(y_tilde / x)
    n = (z + sigma ** 2 * (T - t) / 2) / sigma / jnp.sqrt(T - t)
    p_n = jax.scipy.stats.norm.pdf(n)
    p_z = p_n / (sigma * jnp.sqrt(T - t))
    p_y_tilde = p_z / y_tilde
    p_y = p_y_tilde / y_scale
    return jnp.log(p_y).sum()


@jax.jit
def stein_Matern(x, y, l, d_log_px, d_log_py):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :param d_log_px: N*D
    :param d_log_py: M*D
    :return: N*M
    """
    K = my_Matern(x, y, l)
    dx_K = dx_Matern(x, y, l)
    dy_K = dy_Matern(x, y, l)
    dxdy_K = dxdy_Matern(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = (d_log_py[None, :] * dx_K).sum(-1)
    part3 = (d_log_px[:, None, :] * dy_K).sum(-1)
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


@jax.jit
def stein_Laplace(x, y, l, d_log_px, d_log_py):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :param d_log_px: N*D
    :param d_log_py: M*D
    :return: N*M
    """
    K = my_Laplace(x, y, l)
    dx_K = dx_Laplace(x, y, l)
    dy_K = dy_Laplace(x, y, l)
    dxdy_K = dxdy_Laplace(x, y, l)
    part1 = d_log_px @ d_log_py.T * K
    part2 = d_log_py.T * dx_K
    part3 = d_log_px * dy_K
    part4 = dxdy_K
    return part1 + part2 + part3 + part4


def train(x, y, y_scale, gy, d_log_py, dy_log_py_fn, rng_key, Ky):
    """
    :param y:
    :param gy:
    :param d_log_py:
    :param dy_log_py_fn:
    :param rng_key:
    :return:
    """
    rng_key, _ = jax.random.split(rng_key)
    n = y.shape[0]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    c_init = c = 1.0
    log_l_init = log_l = jnp.log(1.0)
    A_init = A = 1.0 / jnp.sqrt(n)
    opt_state = optimizer.init((log_l_init, c_init, A_init))

    @jax.jit
    def nllk_func(log_l, c, A):
        l = jnp.exp(log_l)
        n = y.shape[0]
        K = A * Ky(y, y, l, d_log_py, d_log_py) + c
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
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
    for _ in range(10000):
        rng_key, _ = jax.random.split(rng_key)
        log_l, c, A, opt_state, nllk_value = step(log_l, c, A, opt_state, rng_key)
        # # Debug code
        # if jnp.isnan(nllk_value):
        #     l = jnp.exp(log_l)
        #     K = A * Ky(y, y, l, d_log_py, d_log_py) + c
        #     K_inv = jnp.linalg.inv(K + eps * jnp.eye(n))
        #     pause = True
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
    # A = jnp.exp(log_A)
    # y_debug = jnp.linspace(20, 160, 100)[:, None] / y_scale
    # d_log_py_debug = dy_log_py_fn(y_debug, x)
    # K_train_train = stein_Laplace(y, y, l, d_log_py, d_log_py) + c
    # K_train_train_inv = jnp.linalg.inv(K_train_train + eps * jnp.eye(n))
    # K_test_train = stein_Laplace(y_debug, y, l, d_log_py_debug, d_log_py) + c
    # gy_debug = K_test_train @ K_train_train_inv @ gy
    # plt.figure()
    # plt.scatter(y * y_scale, gy)
    # plt.plot(y_debug * y_scale, gy_debug)
    # plt.show()
    pause = True
    return jnp.exp(log_l), c, A


class CBQ:
    def __init__(self, kernel_x, kernel_y):
        if kernel_y == 'rbf':
            self.Ky = my_RBF
            self.ly = 0.5
        elif kernel_y == 'matern':
            self.Ky = my_Matern
            self.ly = 0.5
        elif kernel_y == 'laplace':
            self.Ky = my_Laplace
            self.ly = 0.5
        elif kernel_y == 'stein_matern':
            self.Ky = stein_Matern
        elif kernel_y == 'stein_laplace':
            self.Ky = stein_Laplace
        else:
            raise NotImplementedError

        if 'stein' not in kernel_y:
            self.cbq = self.cbq_no_stein
        elif 'stein' in kernel_y:
            self.cbq = self.cbq_stein
        else:
            pass

        if kernel_x == 'rbf':  # This is the best kernel for x
            self.Kx = my_RBF
            self.one_d_Kx = my_RBF
            self.lx = 1.5
        elif kernel_x == 'matern':
            self.Kx = my_Matern
            self.one_d_Kx = my_Matern
            self.lx = 0.7
        else:
            raise NotImplementedError
        return

    # @partial(jax.jit, static_argnums=(0,))
    def cbq_no_stein(self, X, Y, gY, rng_key, sigma):
        Nx = X.shape[0]
        Ny = Y.shape[1]
        eps = 1e-6
        Sigma = jnp.zeros(Nx)
        Mu = jnp.zeros(Nx)
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            Yi_standardized, Yi_scale = finance_utils.scale(Yi)
            gYi = gY[i, :][:, None]
            # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'

            K = self.Ky(Yi_standardized, Yi_standardized, self.ly) + eps * jnp.eye(Ny)
            K_inv = jnp.linalg.inv(K)
            phi = K.mean(1)
            varphi = K.mean()
            mu_standardized = phi.T @ K_inv @ gYi
            std_standardized = jnp.sqrt(varphi - phi.T @ K_inv @ phi)

            Sigma = Sigma.at[i].set(std_standardized.squeeze())
            Mu = Mu.at[i].set(mu_standardized.squeeze())

        return Mu, Sigma

    # @partial(jax.jit, static_argnums=(0,))
    def cbq_stein(self, X, Y, gY, rng_key):
        """
        :param X: X is of size Nx
        :param Y: Y is of size Nx * Ny
        :param gY: gY is g(Y)
        :return: return the expectation E[g(Y)|X=x] of size Nx
        """

        Nx = X.shape[0]
        Ny = Y.shape[1]
        eps = 1e-6
        Sigma = jnp.zeros(Nx)
        Mu = jnp.zeros(Nx)
        for i in range(Nx):
            x = X[i]
            Yi = Y[i, :][:, None]
            Yi_standardized, Yi_scale = finance_utils.scale(Yi)
            gYi = gY[i, :][:, None]

            grad_y_log_py_x_fn = partial(grad_y_log_py_x, sigma=0.3, T=2, t=1, y_scale=Yi_scale)
            dy_log_py_x = grad_y_log_py_x_fn(Yi_standardized, x)
            ly, c, A = train(x, Yi_standardized, Yi_scale, gYi,
                             dy_log_py_x, grad_y_log_py_x_fn, rng_key, self.Ky)
            # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'

            K = A * self.Ky(Yi_standardized, Yi_standardized, ly, dy_log_py_x, dy_log_py_x) + c
            K_inv = jnp.linalg.inv(K + eps * jnp.eye(Ny))
            mu = c * (K_inv @ gYi).sum()
            std = jnp.sqrt(c - K_inv.sum() * c ** 2)

            if mu > 15 or mu < 0:
                mu = gYi.mean()
            Sigma = Sigma.at[i].set(std.squeeze())
            Mu = Mu.at[i].set(mu.squeeze())

            # # Large sample mu
            # print('True value', price(X[i], 10000, rng_key)[1].mean())
            # print(f'MC with {Ny} number of Y', gYi.mean())
            # print(f'BMC with {Ny} number of Y', mu)
            # print(f"=================")
            # pause = True
        return Mu, Sigma

    @partial(jax.jit, static_argnums=(0,))
    def GP(self, psi_y_x_mean, psi_y_x_std, X, X_prime):
        """
        :param psi_y_x_mean: Nx * 1
        :param psi_y_x_std: Nx * 1
        :param X: Nx * 1
        :param X_prime: N_prime * 1
        :return:
        """
        Nx = psi_y_x_mean.shape[0]
        Mu_standardized, Mu_mean, Mu_std = finance_utils.standardize(psi_y_x_mean)
        Sigma_standardized = psi_y_x_std / Mu_std
        X_standardized, X_mean, X_std = finance_utils.standardize(X)
        X_prime_standardized = (X_prime - X_mean) / X_std
        noise = 0.001

        K_train_train = self.Kx(X_standardized, X_standardized, self.lx) + noise * jnp.eye(Nx)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_test_train = self.Kx(X_prime_standardized, X_standardized, self.lx)
        K_test_test = self.Kx(X_prime_standardized, X_prime_standardized, self.lx) + noise
        mu_y_x_prime = K_test_train @ K_train_train_inv @ Mu_standardized
        var_y_x_prime = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
        std_y_x_prime = jnp.sqrt(var_y_x_prime)

        mu_y_x_prime_original = mu_y_x_prime * Mu_std + Mu_mean
        # TODO: Adding jnp.mean(Sigma_standardized) is a bit suspicious here.
        std_y_x_prime_original = std_y_x_prime * Mu_std + jnp.mean(psi_y_x_std)
        return mu_y_x_prime_original, std_y_x_prime_original

    # GP for debugging purposes, not it can only run without jax.jit
    def GP_debug(self, psi_y_x_mean, psi_y_x_std, X, ny):
        Nx = psi_y_x_mean.shape[0]
        Mu_standardized, Mu_mean, Mu_std = finance_utils.standardize(psi_y_x_mean)
        Sigma_standardized = psi_y_x_std / Mu_std
        X_standardized, X_mean, X_std = finance_utils.standardize(X)
        noise = 0.001
        x_debug = jnp.linspace(20, 120, 100)[:, None]
        x_debug_standardized = (x_debug - X_mean) / X_std

        K_train_train = self.Kx(X_standardized, X_standardized, self.lx) + noise * jnp.eye(Nx)
        K_train_train_inv = jnp.linalg.inv(K_train_train)
        K_train_debug = self.Kx(X_standardized, x_debug_standardized, self.lx)
        mu_y_x_debug = K_train_debug.T @ K_train_train_inv @ Mu_standardized
        var_y_x_debug = self.Kx(x_debug_standardized, x_debug_standardized,
                                self.lx) + noise - K_train_debug.T @ K_train_train_inv @ K_train_debug
        std_y_x_debug = jnp.sqrt(jnp.diag(var_y_x_debug))
        mu_y_x_debug_original = mu_y_x_debug * Mu_std + Mu_mean
        # TODO: Adding jnp.mean(Sigma_standardized) is a bit suspicious here.
        std_y_x_debug_original = std_y_x_debug * Mu_std + jnp.mean(psi_y_x_std)

        true_X = jnp.load(f"{args.save_path}/finance_X.npy")
        true_EgY_X = jnp.load(f"{args.save_path}/finance_EgY_X.npy")

        plt.figure()
        plt.plot(x_debug.squeeze(), mu_y_x_debug_original.squeeze(), color='blue', label='predict')
        plt.plot(true_X, true_EgY_X, color='red', label='true')
        plt.scatter(X.squeeze(), psi_y_x_mean.squeeze())
        plt.fill_between(x_debug.squeeze(), mu_y_x_debug_original.squeeze() - std_y_x_debug_original,
                         mu_y_x_debug_original.squeeze() + std_y_x_debug_original, color='blue', alpha=0.2)
        plt.legend()
        plt.title(f"GP_finance_X_{Nx}_y_{ny}")
        plt.savefig(f"{args.save_path}/figures/GP_finance_X_{Nx}_y_{ny}.pdf")
        # plt.show()
        plt.close()
        pause = True
        return

    def debug(self, Nx, Ny, psi_x_mean, St, St_prime, mu_y_x_prime_cbq, std_y_x_prime_cbq,
                    mean_shrinkage_mean, mu_y_x_prime_IS, mu_y_x_prime_poly):
        true_X = jnp.load(f"{args.save_path}/finance_X.npy")
        true_EgY_X = jnp.load(f"{args.save_path}/finance_EgY_X.npy")

        plt.figure()
        plt.ylim(-2, 15)
        plt.plot(true_X, true_EgY_X, color='red', label='true')
        plt.scatter(St.squeeze(), psi_x_mean.squeeze())
        plt.plot(St_prime.squeeze(), mu_y_x_prime_cbq.squeeze(), color='blue', label='BMC')
        plt.plot(St_prime.squeeze(), mu_y_x_prime_IS.squeeze(), color='green', label='IS')
        plt.plot(St_prime.squeeze(), mu_y_x_prime_poly.squeeze(), color='orange', label='poly')
        plt.plot(St_prime.squeeze(), mean_shrinkage_mean.squeeze(), color='purple', label='mean shrinkage')
        plt.legend()
        plt.title(f"GP_finance_X_{Nx}_y_{Ny}")
        plt.savefig(f"{args.save_path}/figures/GP_finance_X_{Nx}_y_{Ny}.pdf")
        # plt.show()
        plt.close()

        jnp.save(f"{args.save_path}/BMC_samples_X_{Nx}_y_{Ny}.npy", psi_x_mean)
        jnp.save(f"{args.save_path}/BMC_mean_X_{Nx}_y_{Ny}.npy", mu_y_x_prime_cbq)
        jnp.save(f"{args.save_path}/BMC_std_X_{Nx}_y_{Ny}.npy", std_y_x_prime_cbq)
        jnp.save(f"{args.save_path}/IS_mean_X_{Nx}_y_{Ny}.npy", mu_y_x_prime_IS)
        jnp.save(f"{args.save_path}/poly_mean_X_{Nx}_y_{Ny}.npy", mu_y_x_prime_poly)
        jnp.save(f"{args.save_path}/mean_shrinkage_mean_X_{Nx}_y_{Ny}.npy", mean_shrinkage_mean)
        pause = True
        return


@partial(jax.jit, static_argnums=(1,))
def price(St, N, rng_key):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    K1 = 50
    K2 = 150
    s = -0.2
    sigma = 0.3
    T = 2
    t = 1

    output_shape = (St.shape[0], N)
    rng_key, _ = jax.random.split(rng_key)
    epsilon = jax.random.normal(rng_key, shape=output_shape)
    ST = St * jnp.exp(sigma * jnp.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = jnp.maximum(ST - K1, 0) + jnp.maximum(ST - K2, 0) - 2 * jnp.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = jnp.maximum((1 + s) * ST - K1, 0) + jnp.maximum((1 + s) * ST - K2, 0) - 2 * jnp.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)
    return ST, psi_ST_1 - psi_ST_2


def save_true_value(args):
    seed = args.seed
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
    _, loss = price(St, 100000, rng_key)
    St = St.squeeze()
    ind = jnp.argsort(St)
    value = loss.mean(1)
    jnp.save(f"{args.save_path}/finance_X.npy", St[ind])
    jnp.save(f"{args.save_path}/finance_EgY_X.npy", value[ind])
    plt.figure()
    plt.plot(St[ind], value[ind])
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\mathbb{E}[g(Y) \mid X]$")
    plt.title("True value for finance experiment")
    plt.savefig(f"{args.save_path}/true_distribution.pdf")
    # plt.show()
    # plt.close()
    return


def cbq_option_pricing(args):
    seed = args.seed
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
    Nx_array = [5, 10]
    # Nx_array = [3, 5, 10, 20, 30]
    Ny_array = [10, 30, 50]
    # Ny_array = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cbq_mean_dict = {}
    cbq_std_dict = {}
    poly_mean_dict = {}
    poly_std_dict = {}
    IS_mean_dict = {}
    IS_std_dict = {}
    mean_shrinkage_mean_dict = {}
    mean_shrinkage_std_dict = {}
    MC_list = []

    St_prime = jnp.linspace(20., 100., 100)[:, None]
    test_ind = 50
    St_prime_single = St_prime[test_ind][:, None]
    rng_key, _ = jax.random.split(rng_key)
    true_value = price(St_prime_single, 100000, rng_key)[1].mean()
    print('True Value is:', true_value)

    kernel_x = args.kernel_x
    kernel_y = args.kernel_y
    CBQ_class = CBQ(kernel_x=kernel_x, kernel_y=kernel_y)
    for Nx in Nx_array:
        cbq_mean_array = jnp.array([])
        cbq_std_array = jnp.array([])
        poly_mean_array = jnp.array([])
        poly_std_array = jnp.array([])
        IS_mean_array = jnp.array([])
        IS_std_array = jnp.array([])
        mean_shrinkage_mean_array = jnp.array([])
        mean_shrinkage_std_array = jnp.array([])

        for Ny in tqdm(Ny_array):
            rng_key, _ = jax.random.split(rng_key)
            # epsilon = jax.random.normal(rng_key, shape=(Nx, 1))
            # St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
            St = jnp.linspace(20, 100, Nx)[:, None]
            ST, loss = price(St, Ny, rng_key)

            # St is X, ST is Y, loss is g(Y)
            psi_x_mean, psi_x_std = CBQ_class.cbq(St, ST, loss, rng_key)
            psi_x_std = np.nan_to_num(psi_x_std, nan=0.3)
            mu_y_x_prime_cbq, std_y_x_prime_cbq = CBQ_class.GP(psi_x_mean, psi_x_std, St, St_prime)
            mc_mean = loss.mean(1)[:, None]
            mc_std = 0 * mc_mean
            mean_shrinkage_mean, mean_shrinkage_std = CBQ_class.GP(mc_mean, mc_std, St, St_prime)
            mu_y_x_prime_IS, std_y_x_prime_IS = importance_sampling(py_x_fn, St_prime, St, ST, loss)
            mu_y_x_prime_poly, std_y_x_prime_poly = polynomial(St, ST, loss, St_prime)

            CBQ_class.debug(Nx, Ny, psi_x_mean, St, St_prime, mu_y_x_prime_cbq, std_y_x_prime_cbq,
                            mean_shrinkage_mean, mu_y_x_prime_IS, mu_y_x_prime_poly)

            cbq_mean_array = jnp.append(cbq_mean_array, mu_y_x_prime_cbq[test_ind])
            cbq_std_array = jnp.append(cbq_std_array, jnp.diag(std_y_x_prime_cbq)[test_ind])
            poly_mean_array = jnp.append(poly_mean_array, mu_y_x_prime_poly[test_ind])
            poly_std_array = jnp.append(poly_std_array, std_y_x_prime_poly[test_ind])
            IS_mean_array = jnp.append(IS_mean_array, mu_y_x_prime_IS[test_ind])
            IS_std_array = jnp.append(IS_std_array, std_y_x_prime_IS[test_ind])
            mean_shrinkage_mean_array = jnp.append(mean_shrinkage_mean_array, mean_shrinkage_mean[test_ind])
            mean_shrinkage_std_array = jnp.append(mean_shrinkage_std_array, mean_shrinkage_std[test_ind])

        cbq_mean_dict[f"{Nx}"] = cbq_mean_array
        cbq_std_dict[f"{Nx}"] = cbq_std_array
        poly_mean_dict[f"{Nx}"] = poly_mean_array
        poly_std_dict[f"{Nx}"] = poly_std_array
        IS_mean_dict[f"{Nx}"] = IS_mean_array
        IS_std_dict[f"{Nx}"] = IS_std_array
        mean_shrinkage_mean_dict[f"{Nx}"] = mean_shrinkage_mean_array
        mean_shrinkage_std_dict[f"{Nx}"] = mean_shrinkage_std_array

    for Ny in Ny_array:
        rng_key, _ = jax.random.split(rng_key)
        MC_list.append(price(St_prime_single, Ny, rng_key)[1].mean())

    with open(f"{args.save_path}/BMC_mean", 'wb') as f:
        pickle.dump(cbq_mean_dict, f)
    with open(f"{args.save_path}/BMC_std", 'wb') as f:
        pickle.dump(cbq_std_dict, f)
    with open(f"{args.save_path}/poly", 'wb') as f:
        pickle.dump(poly_mean_dict, f)
    with open(f"{args.save_path}/importance_sampling", 'wb') as f:
        pickle.dump(IS_mean_dict, f)
    with open(f"{args.save_path}/mean_shrinkage", 'wb') as f:
        pickle.dump(mean_shrinkage_mean_dict, f)
    jnp.save(f"{args.save_path}/MC", jnp.array(MC_list))

    fig, axs = plt.subplots(len(Nx_array), 1, figsize=(10, len(Nx_array) * 3))
    for i, ax in enumerate(axs):
        Nx = Nx_array[i]
        axs[i].set_ylim(2, 16)
        axs[i].axhline(y=true_value, linestyle='--', color='black', label='true value')
        axs[i].plot(Ny_array, MC_list, color='b', label='MC')
        axs[i].plot(Ny_array, cbq_mean_dict[f"{Nx}"], color='r', label=f'CBQ Nx = {Nx}')
        axs[i].plot(Ny_array, poly_mean_dict[f"{Nx}"], color='brown', label=f'Polynomial Nx = {Nx}')
        axs[i].plot(Ny_array, IS_mean_dict[f"{Nx}"], color='darkgreen', label=f'IS Nx = {Nx}')
        axs[i].plot(Ny_array, mean_shrinkage_mean_dict[f"{Nx}"], color='darkorange', label=f'Mean Shrinkage Nx = {Nx}')
        axs[i].fill_between(Ny_array, cbq_mean_dict[f"{Nx}"] - 2 * cbq_std_dict[f"{Nx}"],
                            cbq_mean_dict[f"{Nx}"] + 2 * cbq_std_dict[f"{Nx}"], color='r', alpha=0.5)
        axs[i].legend()
        # axs[i].set_xscale('log')
    plt.tight_layout()
    plt.suptitle("Finance Dataset")
    plt.savefig(f"{args.save_path}/figures/all_methods.pdf")
    plt.show()
    # plt.close()
    return


def main(args):
    seed = args.seed
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)

    visualize_brownian = False
    debug_BSM = False
    if visualize_brownian:
        n = 100.
        T = 1.
        dt = T / n
        plt.figure()
        for i in range(10):
            St = finance_utils.Geometric_Brownian(n, dt, rng_key)
            plt.plot(St)
        # plt.show()
    elif debug_BSM:
        finance_utils.BSM_butterfly_analytic()
    else:
        pass
    cbq_option_pricing(args)
    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/finance/'
    args.save_path += f"seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/figures/", exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print(f'Device is {jax.devices()}')
    print(args.seed)
    save_true_value(args)
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")

