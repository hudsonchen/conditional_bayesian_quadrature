import jax
import jax.numpy as jnp
import os
import shutil
import pwd
import jax.scipy
import jax.scipy.stats
import matplotlib.pyplot as plt
import argparse
import baselines
from tqdm import tqdm
from kernels import *
from utils import sensitivity_utils
import time
import optax
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    # os.chdir("/home/zongchen/CBQ")
    os.chdir("/home/zongchen/fx_bayesian_quaduature/CBQ")
    # os.environ[
    #     "XLA_FLAGS"
    # ] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREAD"] = "1"
elif pwd.getpwuid(os.getuid())[0] == 'ucabzc9':
    os.chdir("/home/ucabzc9/Scratch/CBQ")
else:
    pass

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')
    # Args settings
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--g_fn', type=str, default=None)
    parser.add_argument('--qmc', action='store_true', default=False)
    parser.add_argument('--kernel_x', type=str)
    parser.add_argument('--kernel_theta', type=str)
    parser.add_argument('--baseline_use_variance', action='store_true', default=False)
    args = parser.parse_args()
    return args

def generate_data(rng_key, D, N, noise):
    """
    :param rng_key:
    :param D: int
    :param N: int
    :param noise: std for Gaussian likelihood
    :return: Y is N*(D-1), Z is N*1
    """
    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.uniform(rng_key, shape=(N, D - 1), minval=-1.0, maxval=1.0)
    Y_with_one = jnp.hstack([Y, jnp.ones([Y.shape[0], 1])])
    rng_key, _ = jax.random.split(rng_key)
    beta_true = jax.random.normal(rng_key, shape=(D, 1))
    rng_key, _ = jax.random.split(rng_key)
    Z = Y_with_one @ beta_true + jax.random.normal(rng_key, shape=(N, 1)) * noise
    return Y, Z


# @jax.jit
def posterior_full(Y, Z, prior_cov, noise):
    """
    :param prior_cov: (N3, D)
    :param Y: (N_data, D-1)
    :param Z: (N_data, 1)
    :param noise: float
    :return: (N3, D), (N3, D, D)
    """
    Y_with_one = jnp.hstack([Y, jnp.ones([Y.shape[0], 1])])
    D = prior_cov.shape[-1]
    prior_cov_inv = 1. / prior_cov
    # (N3, D, D)
    prior_cov_inv = jnp.einsum('ij,jk->ijk', prior_cov_inv, jnp.eye(D))
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * Y_with_one.T @ Y_with_one)
    post_mean = beta * post_cov @ Y_with_one.T @ Z
    return post_mean.squeeze(), post_cov


# @jax.jit
def normal_logpdf(x, mu, Sigma):
    """
    :param x: (N2, D)
    :param mu: (N3, D)
    :param Sigma: (N3, D, D)
    :return: (N2, N3)
    """
    N2, D = x.shape
    N3 = mu.shape[0]

    x_expanded = jnp.expand_dims(x, 1)  # Shape (N2, 1, D)
    mean_expanded = jnp.expand_dims(mu, 0)  # Shape (1, N3, D)

    diff = x_expanded - mean_expanded  # Shape (N2, N3, D)
    precision_matrix = jnp.linalg.inv(Sigma)  # Shape (N3, D, D)
    exponent = -0.5 * jnp.einsum('nij, njk, nik->ni', diff, precision_matrix, diff)  # Shape (N2, N3)

    normalization = -0.5 * (D * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(Sigma)))  # Shape (N3,)
    normalization = jnp.expand_dims(normalization, 0)  # Shape (1, N3)

    return normalization + exponent  # Shape (N2, N3)


# @jax.jit
def normal_logpdf_vectorized(x, mu, Sigma):
    """
    :param x: (N1, N2, D)
    :param mu: (N3, D)
    :param Sigma: (N3, D, D)
    :return: (N1, N2, N3)
    """
    D = x.shape[-1]
    x_expanded = jnp.expand_dims(x, 2)
    mean_expanded = jnp.expand_dims(mu, (0, 1))
    # covariance_expanded = jnp.expand_dims(covariance, 0)

    diff = x_expanded - mean_expanded
    precision_matrix = jnp.linalg.inv(Sigma)
    exponent = -0.5 * jnp.einsum('nijk, jkl, nijl->nij', diff, precision_matrix, diff)
    normalization = -0.5 * (D * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(jnp.linalg.det(Sigma)))
    return normalization + exponent


# @jax.jit
def posterior_log_llk_vectorized(X, prior_cov_base, Y, Z, theta, noise):
    """
    :param X: (N1, N2, D)
    :param prior_cov_base: scalar
    :param Y: data
    :param Z: data
    :param theta: (N3, D)
    :param noise: scalar
    :return:
    """
    D = X.shape[-1]
    # prior_cov is (N3, D)
    prior_cov = jnp.ones([1, D]) * prior_cov_base + theta
    # post_mean is (N3, D), post_cov is (N3, D, D)
    post_mean, post_cov = posterior_full(Y, Z, prior_cov, noise)
    return normal_logpdf_vectorized(X, post_mean, post_cov)



# @jax.jit
def posterior_log_llk(X, prior_cov_base, Y, Z, theta, noise):
    """
    :param X: (N2, D)
    :param prior_cov_base: scalar
    :param Y: data
    :param Z: data
    :param theta: (D, )
    :param noise: data noise
    :return:
    """
    D = X.shape[-1]
    # Turn theta into shape (N3, D)
    theta = theta[None, :]
    # prior_cov is (N3, D)
    prior_cov = jnp.ones([1, D]) * prior_cov_base + theta
    # post_mean is (N3, D), post_cov is (N3, D, D)
    post_mean, post_cov = posterior_full(Y, Z, prior_cov, noise)
    return normal_logpdf(X, post_mean, post_cov).squeeze()


def score_fn(X, mu, sigma):
    """
    return \nabla_y log p(X|mu, sigma)
    :param X: (N, D)
    :param mu: (D, )
    :param sigma: (D, D)
    :return: (N, D)
    """
    return -(X - mu[None, :]) @ jnp.linalg.inv(sigma)


def g1(X):
    """
    :param y: y is a N * D array
    """
    return X.sum(1)


def g1_ground_truth(mu, Sigma):
    return mu.sum()


def g2(X):
    """
    :param y: y is a N * D array
    """
    D = X.shape[1]
    return 10 * jnp.exp(-0.5 * ((X ** 2).sum(1) / (D ** 1))) + (X ** 2).sum(1)


def g2_ground_truth(mu, Sigma):
    """
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    D = mu.shape[0]
    analytical_1 = jnp.exp(-0.5 * mu.T @ jnp.linalg.inv(jnp.eye(D) * (D ** 1) + Sigma) @ mu)
    analytical_2 = jnp.linalg.det(jnp.eye(D) + Sigma / (D ** 1)) ** (-0.5)
    analytical = analytical_1 * analytical_2
    return 10 * analytical + jnp.diag(Sigma).sum() + mu.T @ mu


def g3(X):
    return (X ** 2).sum(1)


def g3_ground_truth(mu, Sigma):
    return jnp.diag(Sigma).sum() + mu.T @ mu


def g4(X):
    """
    Only for D = 2
    :param y: (N, D)
    :return: (N, )
    """
    pred = jnp.array([0.3, 1.0])
    return X @ pred


def g4_ground_truth(mu, Sigma):
    """
    Only for D = 2
    :param mu: (D, )
    :param Sigma: (D, D)
    :return: scalar
    """
    pred = jnp.array([0.3, 1.0])
    return mu.T @ pred


def Monte_Carlo(gy):
    return gy.mean(0)


# @jax.jit
def Bayesian_Monte_Carlo_RBF(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    :param var_X_theta: (D, D)
    :param mu_X_theta: (D, )
    :param rng_key:
    :param X: (N, D)
    :param f_X: (N, )
    :return:
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    l_array = jnp.array([0.1, 0.6, 1.0, 3.0]) * D
    nll_array = 0 * l_array
    A_list = []

    for i, l in enumerate(l_array):
        K_no_scale = my_RBF(X, X, l)
        A = f_X.T @ K_no_scale @ f_X / N
        A_list.append(A)
        K = A * K_no_scale
        K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
        nll = -(-0.5 * f_X.T @ K_inv @ f_X - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / N
        nll_array = nll_array.at[i].set(nll)

    if D > 2:
        l = l_array[nll_array.argmin()]
        A = A_list[nll_array.argmin()]
    else:
        A = 1.
        l = 1.

    K = A * my_RBF(X, X, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_RBF_Gaussian(mu_X_theta, var_X_theta, l, X)
    varphi = A * kme_double_RBF_Gaussian(mu_X_theta, var_X_theta, l)

    I_BQ_mean = phi.T @ K_inv @ f_X
    I_BQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))
    pause = True
    return I_BQ_mean, I_BQ_std


def Bayesian_Monte_Carlo_RBF_vectorized_on_T_test(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    :param var_X_theta: (T_test, D, D)
    :param mu_X_theta: (T_test, D)
    :param rng_key:
    :param X: (N, D)
    :param f_X: (N, )
    :return:
    """
    # Define a function that takes only the parameters you want to vectorize over
    def single_instance(mu_single, var_single):
        return Bayesian_Monte_Carlo_RBF(rng_key, X, f_X, mu_single, var_single)

    # Use jax.vmap to vectorize over the first dimension of mu_X_theta and var_X_theta
    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(mu_X_theta, var_X_theta)


def Bayesian_Monte_Carlo_RBF_vectorized_on_T(rng_key, X, f_X, mu_X_theta, var_X_theta):
    """
    :param var_X_theta: (T, D, D)
    :param mu_X_theta: (T, D)
    :param rng_key:
    :param X: (T, N, D)
    :param f_X: (T, N)
    :return:
    """
    # Define a function that takes only the parameters you want to vectorize over
    def single_instance(X_single, f_X_single, mu_X_theta_single, var_X_theta_single):
        return Bayesian_Monte_Carlo_RBF(rng_key, X_single, f_X_single, mu_X_theta_single, var_X_theta_single)

    # Use jax.vmap to vectorize over the first dimension of mu_X_theta and var_X_theta
    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(X, f_X, mu_X_theta, var_X_theta)


def Bayesian_Monte_Carlo_Matern_vectorized_on_T(rng_key, u, X, f_X, mu_X_theta, var_X_theta):
    """    
    We only implement this for D = 2.
    :param u: (T, N, D)
    :param var_X_theta: (T, D, D)
    :param mu_X_theta: (T, D)
    :param rng_key:
    :param X: (T, N, D)
    :param f_X: (T, N)
    :return:
    """
    # Define a function that takes only the parameters you want to vectorize over
    def single_instance(u_single, X_single, f_X_single, mu_X_theta_single, var_X_theta_single):
        return Bayesian_Monte_Carlo_Matern(rng_key, u_single, X_single, f_X_single, mu_X_theta_single, var_X_theta_single)

    # Use jax.vmap to vectorize over the first dimension of mu_X_theta and var_X_theta
    vectorized_function = jax.vmap(single_instance)
    return vectorized_function(u, X, f_X, mu_X_theta, var_X_theta)


def Bayesian_Monte_Carlo_Matern(rng_key, u, X, f_X, mu_X_theta, var_X_theta):
    """
    We only implement this for D = 2.
    :param u: (N, D)
    :param var_X_theta: (D, D)
    :param mu_X_theta: (D, )
    :param rng_key:
    :param y: (N, D)
    :param f_X: (N, )
    :return:
    """
    N, D = X.shape[0], X.shape[1]
    eps = 1e-6

    A = 1.
    l = 1.

    u1 = u[:, 0][:, None]
    u2 = u[:, 1][:, None]

    K = A * my_Matern(u1, u1, l) + A * my_Matern(u2, u2, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = A * kme_Matern_Gaussian(l, u1) + A * kme_Matern_Gaussian(l, u2)
    varphi = phi.mean()

    CBQ_mean = phi.T @ K_inv @ f_X
    CBQ_std = jnp.sqrt(jnp.abs(varphi - phi.T @ K_inv @ phi))

    CBQ_mean = CBQ_mean.squeeze()
    CBQ_std = CBQ_std.squeeze()
    pause = True
    return CBQ_mean, CBQ_std



# @jax.jit
def GP(rng_key, I_mean, I_std, X, X_test, eps, kernel_fn):
    """
    :param kernel_fn: Matern or RBF
    :param eps:
    :param I_mean: (T, )
    :param I_std: (T, )
    :param X: (T, D)
    :param X_test: (T_test, D)
    :return:
    """
    T, D = X.shape[0], X.shape[1]
    l_array = jnp.array([0.3, 1.0, 2.0, 3.0]) * D

    sigma_array = jnp.array([0.0])
    A_array = 0 * l_array
    nll_array = jnp.zeros([len(l_array), 1])

    for i, l in enumerate(l_array):
        K_no_scale = kernel_fn(X, X, l)
        A = I_mean.T @ K_no_scale @ I_mean / T
        A_array = A_array.at[i].set(A)
        K = A * kernel_fn(X, X, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
        K_inv = jnp.linalg.inv(K)
        nll = -(-0.5 * I_mean.T @ K_inv @ I_mean - 0.5 * jnp.log(jnp.linalg.det(K) + 1e-6)) / T
        nll_array = nll_array.at[i].set(nll)

    l = l_array[jnp.argmin(nll_array)]
    A = A_array[jnp.argmin(nll_array)]

    K_train_train = A * kernel_fn(X, X, l) + eps * jnp.eye(T) + jnp.diag(I_std ** 2)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = A * kernel_fn(X_test, X, l)
    K_test_test = A * kernel_fn(X_test, X_test, l) + eps * jnp.eye(X_test.shape[0])

    mu_X_theta_test = K_test_train @ K_train_train_inv @ I_mean
    var_X_theta_test = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    var_X_theta_test = jnp.abs(var_X_theta_test)
    std_X_theta_test = jnp.sqrt(var_X_theta_test)
    pause = True
    return mu_X_theta_test, std_X_theta_test


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_cov_base = 2.0
    noise = 1.0
    sample_size = 5000
    T_test = 100
    data_number = 5
    # theta is (N, D-1), X is (N, 1)
    rng_key, _ = jax.random.split(rng_key)
    Y, Z = generate_data(rng_key, D, data_number, noise)

    if args.g_fn == 'g1':
        g = g1
        g_ground_truth_fn = g1_ground_truth
    elif args.g_fn == 'g2':
        g = g2
        g_ground_truth_fn = g2_ground_truth
    elif args.g_fn == 'g3':
        g = g3
        g_ground_truth_fn = g3_ground_truth
    elif args.g_fn == 'g4':
        g = g4
        g_ground_truth_fn = g4_ground_truth
    else:
        raise ValueError('g_fn must be g1 or g2 or g3 or g4!')

    T_array = jnp.array([10, 50, 100])
    # N_array = jnp.array([10, 50, 100])
    N_array = jnp.concatenate((jnp.array([3, 5]), jnp.arange(10, 150, 10)))

    # This is the test point
    Theta_test = jax.random.uniform(rng_key, shape=(T_test, D), minval=-1.0, maxval=1.0)
    prior_cov_test = jnp.array([[prior_cov_base] * D]) + Theta_test
    ground_truth = jnp.zeros(T_test)

    mu_x_theta_test, var_x_theta_test = posterior_full(Y, Z, prior_cov_test, noise)
    # post_mean: (T_test, D), post_var: (T_test, D, D)
    for i in range(T_test):
        ground_truth = ground_truth.at[i].set(g_ground_truth_fn(mu_x_theta_test[i, :], var_x_theta_test[i, :, :]))
    jnp.save(f"{args.save_path}/Theta_test.npy", Theta_test)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    for T in T_array:
        rng_key, _ = jax.random.split(rng_key)
        # This is theta, size T * D
        if args.qmc:
            Theta = sensitivity_utils.qmc_uniform(-1.0, 1.0, D, T)
        else:
            Theta = jax.random.uniform(rng_key, shape=(T, D), minval=-1.0, maxval=1.0)

        for j, N in enumerate(tqdm(N_array)):
            # ======================================== Precompute f(X), X, Theta ========================================
            I_BQ_mean_array = jnp.zeros(T)
            I_BQ_std_array = jnp.zeros(T)
            I_MC_mean_array = jnp.zeros(T)
            I_MC_std_array = jnp.zeros(T)

            # This is X, size T * N * D
            X = jnp.zeros([T, N, D]) + 0.0
            # This is f(X), size T * N
            f_X = jnp.zeros([T, N]) + 0.0
            # This is u, size T * N * D, used for CBQ with Matern kernel
            u_all = jnp.zeros([T, N, D]) + 0.0
            # This is score, size T * N * D, used for CBQ with Stein kernel
            score_all = jnp.zeros([T, N, D]) + 0.0

            prior_cov = jnp.array([[prior_cov_base] * D]) + Theta
            mu_x_theta_all, var_x_theta_all = posterior_full(Y, Z, prior_cov, noise)

            for i in range(T):
                rng_key, _ = jax.random.split(rng_key)
                if args.qmc:
                    X_i, u = sensitivity_utils.qmc_gaussian(mu_x_theta_all[i, :], var_x_theta_all[i, :, :], N)
                    score = score_fn(X_i, mu_x_theta_all[i, :], var_x_theta_all[i, :, :])
                else:
                    u = jax.random.multivariate_normal(rng_key, mean=jnp.zeros_like(mu_x_theta_all[i, :]),
                                                       cov=jnp.diag(jnp.ones_like(mu_x_theta_all[i, :])),
                                                       shape=(N,))
                    L = jnp.linalg.cholesky(var_x_theta_all[i, :, :])
                    X_i = mu_x_theta_all[i, :] + jnp.matmul(L, u.T).T
                    score = score_fn(X_i, mu_x_theta_all[i, :], var_x_theta_all[i, :, :])
                score_all = score_all.at[i, :, :].set(score)
                u_all = u_all.at[i, :, :].set(u)
                X = X.at[i, :, :].set(X_i)
                f_X = f_X.at[i, :].set(g(X_i))
            # ======================================== Precompute f(X), X, Theta ========================================

            # ======================================== Debug code ========================================
            # for i in range(T):
            #     X_i = X[i, :, :]
            #     f_X_i = f_X[i, :]
            #     u_i = u_all[i, :, :]
            #     score_i = score_all[i, :, :]
            #     mu_X_theta_i = mu_x_theta_all[i, :]
            #     var_X_theta_i = var_x_theta_all[i, :, :]

            #     if args.kernel_x == "RBF":
            #         I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_RBF(rng_key, X_i, f_X_i, mu_X_theta_i, var_X_theta_i)
            #     elif args.kernel_x == "Matern":
            #         if D > 2:
            #             raise NotImplementedError("Matern kernel is only implemented for D=2")
            #         I_BQ_mean, I_BQ_std = Bayesian_Monte_Carlo_Matern(rng_key, u_i, X_i, f_X_i, mu_X_theta_i,
            #                                                           var_X_theta_i)
            #     else:
            #         raise NotImplementedError("Kernel not implemented")

            #     I_BQ_mean_array = I_BQ_mean_array.at[i].set(I_BQ_mean)
            #     I_BQ_std_array = I_BQ_std_array.at[i].set(I_BQ_std if not jnp.isnan(I_BQ_std) else 0.01)

            #     I_MC_mean_array = I_MC_mean_array.at[i].set(f_X_i.mean())
            #     I_MC_std_array = I_MC_std_array.at[i].set(f_X_i.std())

            #     true_value = g_ground_truth_fn(mu_X_theta_i, var_X_theta_i)
            #     CBQ_value = I_BQ_mean
            #     print("=============")
            #     print('True value', true_value)
            #     print(f'MC with N={N}', MC_value)
            #     print(f'CBQ with N={N}', CBQ_value)
            #     print(f'CBQ uncertainty {I_BQ_std}')
            #     print(f"=============")
            #     pause = True
            #     ======================================== Debug code ========================================

            # ======================================== CBQ ========================================
            time0 = time.time()
            if args.kernel_x == "RBF":
                I_BQ_mean_array, I_BQ_std_array = Bayesian_Monte_Carlo_RBF_vectorized_on_T(rng_key, X, f_X, mu_x_theta_all, var_x_theta_all)
            elif args.kernel_x == "Matern":
                if D > 2:
                    raise NotImplementedError("Matern kernel is only implemented for D=2")
                I_BQ_mean_array, I_BQ_std_array = Bayesian_Monte_Carlo_Matern_vectorized_on_T(rng_key, u, X, f_X, mu_x_theta_all, var_x_theta_all)
            else:
                raise NotImplementedError("Kernel not implemented")
            time_CBQ = time.time() - time0 

            rng_key, _ = jax.random.split(rng_key)
            _, _ = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_Matern)
            time0 = time.time()
            if args.kernel_theta == "RBF":
                CBQ_mean, CBQ_std = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_RBF)
            elif args.kernel_theta == "Matern":
                CBQ_mean, CBQ_std = GP(rng_key, I_BQ_mean_array, I_BQ_std_array, Theta, Theta_test, eps=I_BQ_std_array.mean(), kernel_fn=my_Matern)
            else:
                raise NotImplementedError(f"Unknown kernel {args.kernel_theta}")
            time_CBQ += time.time() - time0
            # ======================================== CBQ ========================================

            # ======================================== KMS ========================================
            time0 = time.time()
            I_MC_mean_array = f_X.mean(1)
            I_MC_std_array = f_X.std(1)
            if args.baseline_use_variance:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, I_MC_std_array, Theta, Theta_test, eps=0., kernel_fn=my_RBF)
            else:
                KMS_mean, KMS_std = baselines.kernel_mean_shrinkage(rng_key, I_MC_mean_array, None, Theta, Theta_test, eps=0., kernel_fn=my_RBF)
            time_KMS = time.time() - time0
            # ======================================== KMS ========================================

            # ======================================== BQ ========================================
            time0 = time.time()
            rng_key, _ = jax.random.split(rng_key)
            BQ_mean, BQ_std = Bayesian_Monte_Carlo_RBF_vectorized_on_T_test(rng_key, X.reshape([N * T, D]), f_X.reshape([N * T]), 
                                            mu_x_theta_test, var_x_theta_test)
            time_BQ = time.time() - time0
            # ======================================== BQ ========================================

            # ======================================== LSMC ========================================
            time0 = time.time()
            if args.baseline_use_variance:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta, X, f_X, Theta_test, baseline_use_variance=True)
            else:
                LSMC_mean, LSMC_std = baselines.polynomial(Theta, X, f_X, Theta_test, baseline_use_variance=False)
            time_LSMC = time.time() - time0
            # ======================================== LSMC ========================================

            # ======================================== IS ========================================
            time0 = time.time()
            # This is unvectorized version of importance sampling
            # log_px_theta_fn = partial(posterior_log_llk, Y=Y, Z=Z, noise=noise, prior_cov_base=prior_cov_base)
            # IS_mean, IS_std = baselines.importance_sampling(log_px_theta_fn, Theta_test, Theta, X, f_X)

            # This is vectorized version of importance sampling
            log_px_theta_fn_vectorized = partial(posterior_log_llk_vectorized, Y=Y, Z=Z, noise=noise, prior_cov_base=prior_cov_base)
            IS_mean, IS_std = baselines.importance_sampling_sensitivity(log_px_theta_fn_vectorized, Theta_test, Theta, X, f_X)
            time_IS = time.time() - time0
            # ======================================== IS ========================================

            rmse_CBQ, rmse_BQ, rmse_KMS, rmse_LSMC, rmse_IS = sensitivity_utils.compute_rmse(ground_truth, CBQ_mean, BQ_mean, KMS_mean, LSMC_mean, IS_mean)
                                                                                     
            calibration = sensitivity_utils.calibrate(ground_truth, CBQ_mean, jnp.diag(CBQ_std))

            sensitivity_utils.save(args, T, N, rmse_CBQ, rmse_BQ, rmse_KMS, rmse_LSMC, rmse_IS,
                                   time_CBQ, time_BQ, time_KMS, time_LSMC, time_IS, calibration)

    return


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity_conjugate/'
    args.save_path += f"seed_{args.seed}__dim_{args.dim}__function_{args.g_fn}__Kx_{args.kernel_x}__Ktheta_{args.kernel_theta}__qmc_{args.qmc}__usevar_{args.baseline_use_variance}"
    os.makedirs(args.save_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = get_config()
    create_dir(args)
    print(f'Device is {jax.devices()}')
    print(f'Seed is {args.seed}')
    main(args)
    save_path = args.save_path
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    if os.path.exists(f"{save_path}__complete"):
        print(f"Removing old results at {save_path}__complete")
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
