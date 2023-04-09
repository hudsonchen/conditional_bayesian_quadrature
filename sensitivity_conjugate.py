import jax
import jax.numpy as jnp
import os
import shutil
import pwd
import jax.scipy
import jax.scipy.stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sensitivity_baselines import *
from tqdm import tqdm
from kernels import *
from utils import finance_utils, sensitivity_utils
import time
import optax
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
    os.environ[
        "XLA_FLAGS"
    ] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREAD"] = "1"
else:
    pass

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=False)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def generate_data(rng_key, D, N, noise):
    """
    :param rng_key:
    :param D: int
    :param N: int
    :param noise: std for Gaussian likelihood
    :return: X is N*(D-1), Y is N*1
    """
    rng_key, _ = jax.random.split(rng_key)
    X = jax.random.uniform(rng_key, shape=(N, D - 1), minval=-1.0, maxval=1.0)
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    rng_key, _ = jax.random.split(rng_key)
    beta_true = jax.random.normal(rng_key, shape=(D, 1))
    rng_key, _ = jax.random.split(rng_key)
    Y = X_with_one @ beta_true + jax.random.normal(rng_key, shape=(N, 1)) * noise
    return X, Y


@jax.jit
def posterior_full(X, Y, prior_cov, noise):
    """
    :param prior_cov: D*1 array
    :param X: N*(D-1) array
    :param Y: N*1 array
    :param noise: float
    :return:
    """
    X_with_one = jnp.hstack([X, jnp.ones([X.shape[0], 1])])
    prior_cov = jnp.diag(prior_cov.squeeze())
    prior_cov_inv = jnp.diag(1. / prior_cov.squeeze())
    beta_inv = noise ** 2
    beta = 1. / beta_inv
    post_cov = jnp.linalg.inv(prior_cov_inv + beta * X_with_one.T @ X_with_one)
    post_mean = beta * post_cov @ X_with_one.T @ Y
    return post_mean.squeeze(), post_cov


@jax.jit
def normal_logpdf(x, mu, Sigma):
    """
    :param x: (D, )
    :param mu: (D, )
    :param Sigma: (D, D)
    :return:
    """
    D = x.shape[0]
    return jnp.log(2 * jnp.pi) * (-D / 2) + jnp.log(jnp.linalg.det(Sigma)) * (-0.5) - 0.5 * (x - mu).T @ jnp.linalg.inv(Sigma) @ (x - mu)


@jax.jit
def posterior_log_llk(theta, prior_cov_base, X, Y, alpha, noise):
    """
    :param theta: (N, D)
    :param prior_cov_base: scalar
    :param X: data
    :param Y: data
    :param alpha: (D, 1)
    :param noise: scalar
    :return:
    """
    D = theta.shape[1]
    prior_cov = jnp.array([[prior_cov_base] * D]).T + alpha
    post_mean, post_cov = posterior_full(X, Y, prior_cov, noise)
    logpdf = partial(normal_logpdf, mu=post_mean, Sigma=post_cov)
    logpdf_vmap = jax.vmap(logpdf)
    return logpdf_vmap(theta)


def g1(y):
    """
    :param y: y is a N * D array
    """
    return y.sum(1)


def g1_ground_truth(mu, Sigma):
    return mu.sum()


def g2(y):
    """
    :param y: y is a N * D array
    """
    return jnp.exp(-0.5 * ((y ** 2).sum(1)))


def g2_ground_truth(mu, Sigma):
    D = mu.shape[0]
    analytical_1 = jnp.exp(-0.5 * mu.T @ jnp.linalg.inv(jnp.eye(D) + Sigma) @ mu)
    analytical_2 = jnp.linalg.det(jnp.eye(D) + Sigma) ** (-0.5)
    analytical = analytical_1 * analytical_2
    return analytical


def Monte_Carlo(gy):
    return gy.mean(0)


@partial(jax.jit, static_argnames=['Ky'])
def nllk_func(log_l, A, y, gy, Ky, eps):
    N = y.shape[0]
    l = jnp.exp(log_l)
    K = A * Ky(y, y, l) + jnp.eye(N)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    nll = -(-0.5 * gy.T @ K_inv @ gy - 0.5 * jnp.log(jnp.linalg.det(K) + eps)) / N
    return nll


@partial(jax.jit, static_argnames=['optimizer', 'Ky'])
def step(log_l, A, opt_state, optimizer, y, gy, Ky, eps):
    nllk_value, grads = jax.value_and_grad(nllk_func, argnums=(0, 1))(log_l, A, y, gy, Ky, eps)
    updates, opt_state = optimizer.update(grads, opt_state, (log_l, A))
    log_l, A = optax.apply_updates((log_l, A), updates)
    return log_l, A, opt_state, nllk_value


# @jax.jit
def Bayesian_Monte_Carlo(rng_key, y, gy, mu_y_x, sigma_y_x):
    """
    :param rng_key:
    :param y: (N, D)
    :param gy: (N, )
    :return:
    """
    N, D = y.shape[0], y.shape[1]
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    eps = 1e-6

    log_l_init = log_l = jnp.log(1.0)
    A_init = A = 1.0
    opt_state = optimizer.init((log_l_init, A_init))
    Ky = my_RBF

    # Debug code
    # l_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    # for _ in range(0):
    #     rng_key, _ = jax.random.split(rng_key)
    #     log_l, A, opt_state, nllk_value = step(log_l, A, opt_state, optimizer, y, gy, Ky, eps)
        # # Debug code
    #     l_debug_list.append(jnp.exp(log_l))
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3 = fig.subplots(1, 3)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(A_debug_list)
    # ax_3.plot(nll_debug_list)
    # plt.show()

    l = jnp.exp(log_l)
    K = my_RBF(y, y, l)
    K_inv = jnp.linalg.inv(K + eps * jnp.eye(N))
    phi = kme_RBF_Gaussian(mu_y_x, sigma_y_x, l, y)
    varphi = kme_double_RBF_Gaussian(mu_y_x, sigma_y_x, l)

    BMC_mean = phi.T @ K_inv @ gy
    BMC_std = jnp.sqrt(varphi - phi.T @ K_inv @ phi)
    pause = True
    return BMC_mean, BMC_std


# @jax.jit
def GP(rng_key, psi_y_x_mean, psi_y_x_std, X, X_prime):
    """
    :param psi_y_x_mean: (n_alpha, )
    :param psi_y_x_std: (n_alpha, )
    :param X: (n_alpha, D)
    :param X_prime: (N_test, D)
    :return:
    """
    Nx = psi_y_x_mean.shape[0]
    eps = 0.01
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)

    log_l_init = log_l = jnp.log(3.0)
    A_init = A = 3.0
    opt_state = optimizer.init((log_l_init, A_init))
    Ky = my_RBF

    # # Debug code
    # l_debug_list = []
    # A_debug_list = []
    # nll_debug_list = []
    for _ in range(100):
        rng_key, _ = jax.random.split(rng_key)
        log_l, A, opt_state, nllk_value = step(log_l, A, opt_state, optimizer, X, psi_y_x_mean, Ky, eps)
        A = A_init
        # Debug code
    #     l_debug_list.append(jnp.exp(log_l))
    #     A_debug_list.append(A)
    #     nll_debug_list.append(nllk_value)
    # fig = plt.figure(figsize=(15, 6))
    # ax_1, ax_2, ax_3 = fig.subplots(1, 3)
    # ax_1.plot(l_debug_list)
    # ax_2.plot(A_debug_list)
    # ax_3.plot(nll_debug_list)
    # plt.show()

    l = jnp.exp(log_l)
    K_train_train = my_RBF(X, X, l) + eps * jnp.eye(Nx)
    K_train_train_inv = jnp.linalg.inv(K_train_train)
    K_test_train = my_RBF(X_prime, X, l)
    K_test_test = my_RBF(X_prime, X_prime, l) + eps
    mu_y_x = K_test_train @ K_train_train_inv @ psi_y_x_mean
    var_y_x = K_test_test - K_test_train @ K_train_train_inv @ K_test_train.T
    std_y_x = jnp.sqrt(var_y_x)
    pause = True
    return mu_y_x, std_y_x


def main(args):
    seed = args.seed
    rng_key = jax.random.PRNGKey(seed)
    D = args.dim
    prior_cov_base = 2.5
    noise = 1.0
    sample_size = 1000
    test_num = 100
    data_number = D + 5
    X, Y = generate_data(rng_key, D, data_number, noise)

    if args.g_fn == 'g1':
        g = g1
        g_ground_truth_fn = g1_ground_truth
    elif args.g_fn == 'g2':
        g = g2
        g_ground_truth_fn = g2_ground_truth
    else:
        raise ValueError('g_fn must be g1 or g2')

    N_alpha_array = jnp.array([5, 10, 20, 50])
    # N_alpha_array = jnp.arange(2, 50, 2)
    # N_theta_array = jnp.array([5, 10, 20])
    N_theta_array = jnp.arange(2, 20)
    # N_theta_array = jnp.arange(2, 50)

    # This is the test point
    alpha_test_line = jax.random.uniform(rng_key, shape=(test_num, D), minval=-2.0, maxval=2.0)
    cov_test_line = jnp.array([[prior_cov_base] * D]) + alpha_test_line
    post_mean_test_line, post_var_test_line = jnp.zeros([test_num, D]), jnp.zeros([test_num, D, D])
    ground_truth = jnp.zeros(test_num)
    for i in range(test_num):
        post_mean, post_var = posterior_full(X, Y, cov_test_line[i, :], noise)
        post_mean_test_line = post_mean_test_line.at[i, :].set(post_mean)
        post_var_test_line = post_var_test_line.at[i, :, :].set(post_var)
        ground_truth = ground_truth.at[i].set(g_ground_truth_fn(post_mean, post_var))
    jnp.save(f"{args.save_path}/test_line.npy", alpha_test_line)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    for n_alpha in N_alpha_array:
        rng_key, _ = jax.random.split(rng_key)
        # This is X, size n_alpha * D
        alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)

        # This is Y, size n_alpha * sample_size * D
        samples_all = jnp.zeros([n_alpha, sample_size, D])
        # This is g(Y), size n_alpha * sample_size
        g_samples_all = jnp.zeros([n_alpha, sample_size])
        mu_y_x_all = jnp.zeros([n_alpha, D])
        var_y_x_all = jnp.zeros([n_alpha, D, D])

        for i in range(n_alpha):
            rng_key, _ = jax.random.split(rng_key)
            prior_cov = jnp.array([[prior_cov_base] * D]).T + alpha_all[i, :]
            mu_y_x, var_y_x = posterior_full(X, Y, prior_cov, noise)
            samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x, cov=var_y_x, shape=(1000, ))
            samples_all = samples_all.at[i, :, :].set(samples)
            g_samples_all = g_samples_all.at[i, :].set(g(samples))
            mu_y_x_all = mu_y_x_all.at[i, :].set(mu_y_x)
            var_y_x_all = var_y_x_all.at[i, :, :].set(var_y_x)

        mse_BMC_array = jnp.zeros(len(N_theta_array))
        mse_KMS_array = jnp.zeros(len(N_theta_array))
        mse_LSMC_array = jnp.zeros(len(N_theta_array))
        mse_IS_array = jnp.zeros(len(N_theta_array))

        time_BMC_array = jnp.zeros(len(N_theta_array))
        time_KMS_array = jnp.zeros(len(N_theta_array))
        time_LSMC_array = jnp.zeros(len(N_theta_array))
        time_IS_array = jnp.zeros(len(N_theta_array))

        for j, n_theta in enumerate(tqdm(N_theta_array)):
            psi_mean_array = jnp.array([])
            psi_std_array = jnp.array([])
            mc_mean_array = jnp.array([])

            t0 = time.time()
            for i in range(n_alpha):
                rng_key, _ = jax.random.split(rng_key)
                samples_i = samples_all[i, :n_theta, :]
                g_samples_i = g_samples_all[i, :n_theta]
                g_samples_i_standardized, g_samples_i_scale = sensitivity_utils.scale(g_samples_i)
                mu_y_x_i = mu_y_x_all[i, :]
                var_y_x_i = var_y_x_all[i, :, :]

                psi_mean, psi_std = Bayesian_Monte_Carlo(rng_key, samples_i, g_samples_i_standardized, mu_y_x_i, var_y_x_i)

                psi_mean_array = jnp.append(psi_mean_array, psi_mean * g_samples_i_scale)
                psi_std_array = jnp.append(psi_std_array, psi_std * g_samples_i_scale)

                MC_value = g_samples_i.mean()
                mc_mean_array = jnp.append(mc_mean_array, MC_value)

                # # Debug
                # true_value = g_ground_truth_fn(mu_y_x_i, var_y_x_i)
                # BMC_value = psi_mean * g_samples_i_scale
                # print("=============")
                # print('True value', true_value)
                # print(f'MC with {n_theta} number of Y', MC_value)
                # print(f'BMC with {n_theta} number of Y', BMC_value)
                # print(f"=============")
                # pause = True

            rng_key, _ = jax.random.split(rng_key)
            BMC_mean, BMC_std = GP(rng_key, psi_mean_array, psi_std_array, alpha_all, alpha_test_line)
            time_BMC = time.time() - t0
            time_BMC_array = time_BMC_array.at[j].set(time_BMC)

            rng_key, _ = jax.random.split(rng_key)
            t0 = time.time()
            KMS_mean, KMS_std = GP(rng_key, mc_mean_array, mc_mean_array * 0, alpha_all, alpha_test_line)
            time_KMS = time.time() - t0
            time_KMS_array = time_KMS_array.at[j].set(time_KMS)

            t0 = time.time()
            LSMC_mean, LSMC_std = polynomial(alpha_all, samples_all[:, :n_theta, :],
                                             g_samples_all[:, :n_theta], alpha_test_line)
            time_LSMC = time.time() - t0
            time_LSMC_array = time_LSMC_array.at[j].set(time_LSMC)

            t0 = time.time()
            log_py_x_fn = partial(posterior_log_llk, X=X, Y=Y, noise=noise, prior_cov_base=prior_cov_base)
            IS_mean, IS_std = importance_sampling(log_py_x_fn, alpha_all, samples_all[:, :n_theta, :],
                                                  g_samples_all[:, :n_theta], alpha_test_line)
            time_IS = time.time() - t0
            time_IS_array = time_IS_array.at[j].set(time_IS)

            sensitivity_utils.save(args, n_alpha, n_theta, BMC_mean, BMC_std, KMS_mean, KMS_std, LSMC_mean,
                                   LSMC_std, IS_mean, IS_std, ground_truth, time_BMC, time_KMS, time_LSMC, time_IS)

            mse_BMC = jnp.mean((BMC_mean - ground_truth) ** 2)
            mse_KMS = jnp.mean((KMS_mean - ground_truth) ** 2)
            mse_LSMC = jnp.mean((LSMC_mean - ground_truth) ** 2)
            mse_IS = jnp.mean((IS_mean - ground_truth) ** 2)

            mse_BMC_array = mse_BMC_array.at[j].set(mse_BMC)
            mse_KMS_array = mse_KMS_array.at[j].set(mse_KMS)
            mse_LSMC_array = mse_LSMC_array.at[j].set(mse_LSMC)
            mse_IS_array = mse_IS_array.at[j].set(mse_IS)

            # Debug
            # print(f"=============")
            # print(f"MSE of BMC with {n_alpha} number of X and {n_theta} number of Y", mse_BMC)
            # print(f"MSE of KMS with {n_alpha} number of X and {n_theta} number of Y", mse_KMS)
            # print(f"MSE of LSMC with {n_alpha} number of X and {n_theta} number of Y", mse_LSMC)
            # print(f"MSE of IS with {n_alpha} number of X and {n_theta} number of Y", mse_IS)
            # print(f"=============")
            # pause = True

        # plotting
        # Debug code
        fig = plt.figure(figsize=(15, 6))
        ax_1, ax_2 = fig.subplots(1, 2)
        ax_1.plot(N_theta_array, mse_BMC_array, label='BMC')
        ax_1.plot(N_theta_array, mse_KMS_array, label='KMS')
        ax_1.plot(N_theta_array, mse_LSMC_array, label='LSMC')
        ax_1.plot(N_theta_array, mse_IS_array, label='IS')
        ax_1.legend()
        ax_1.set_xlabel('Number of Y')
        ax_1.set_yscale('log')
        ax_1.set_ylabel('MSE')
        ax_2.plot(N_theta_array, time_BMC_array, label='BMC')
        ax_2.plot(N_theta_array, time_KMS_array, label='KMS')
        ax_2.plot(N_theta_array, time_LSMC_array, label='LSMC')
        ax_2.plot(N_theta_array, time_IS_array, label='IS')
        ax_2.legend()
        ax_2.set_xlabel('Number of Y')
        ax_2.set_ylabel('Time')

        plt.suptitle(f"Sensitivity analysis with {n_alpha} number of X")
        plt.savefig(f"{args.save_path}/sensitivity_conjugate_Nx_{n_alpha}.pdf")
        plt.show()
        plt.close()

    # For very very large Nx and Ny. Test the performance of other methods
    n_alpha = 1000
    n_theta = 1000
    rng_key, _ = jax.random.split(rng_key)
    # This is X, size n_alpha * D
    alpha_all = jax.random.uniform(rng_key, shape=(n_alpha, D), minval=-1.0, maxval=1.0)

    # This is Y, size n_alpha * n_theta * D
    samples_all = jnp.zeros([n_alpha, n_theta, D])
    # This is g(Y), size n_alpha * sample_size
    g_samples_all = jnp.zeros([n_alpha, n_theta])
    mu_y_x_all = jnp.zeros([n_alpha, D])
    var_y_x_all = jnp.zeros([n_alpha, D, D])

    for i in range(n_alpha):
        rng_key, _ = jax.random.split(rng_key)
        prior_cov = jnp.array([[prior_cov_base] * D]).T + alpha_all[i, :]
        mu_y_x, var_y_x = posterior_full(X, Y, prior_cov, noise)
        samples = jax.random.multivariate_normal(rng_key, mean=mu_y_x, cov=var_y_x, shape=(n_theta, ))
        samples_all = samples_all.at[i, :, :].set(samples)
        g_samples_all = g_samples_all.at[i, :].set(g(samples))
        mu_y_x_all = mu_y_x_all.at[i, :].set(mu_y_x)
        var_y_x_all = var_y_x_all.at[i, :, :].set(var_y_x)

    mc_mean_array = g_samples_all.mean(axis=1)
    rng_key, _ = jax.random.split(rng_key)
    t0 = time.time()
    KMS_mean, KMS_std = GP(rng_key, mc_mean_array, mc_mean_array * 0, alpha_all, alpha_test_line)
    time_KMS_large = time.time() - t0

    t0 = time.time()
    LSMC_mean, LSMC_std = polynomial(alpha_all, samples_all, g_samples_all, alpha_test_line)
    time_LSMC_large = time.time() - t0
    sensitivity_utils.save_large(args, n_alpha, n_theta, KMS_mean, KMS_std, LSMC_mean, LSMC_std,
                                 ground_truth, time_KMS_large, time_LSMC_large)
    # Debug code
    print(f"=============")
    mse_KMS = jnp.mean((KMS_mean - ground_truth) ** 2)
    mse_LSMC = jnp.mean((LSMC_mean - ground_truth) ** 2)
    print(f"KMS mse with {n_alpha} number of X and {n_theta} number of Y", mse_KMS)
    print(f"KMS time with {n_alpha} number of X and {n_theta} number of Y", time_KMS_large)
    print(f"LSMC mse with {n_alpha} number of X and {n_theta} number of Y", mse_LSMC)
    print(f"LSMC time with {n_alpha} number of X and {n_theta} number of Y", time_LSMC_large)
    print(f"=============")
    return


def get_config():
    parser = argparse.ArgumentParser(description='Conditional Bayesian Quadrature for Bayesian sensitivity analysis')
    # Args settings
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--g_fn', type=str, default=None)
    args = parser.parse_args()
    return args


def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f'results/sensitivity_conjugate/'
    args.save_path += f"seed_{args.seed}__dim_{args.dim}"
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
        shutil.rmtree(f"{save_path}__complete")
    os.rename(save_path, f"{save_path}__complete")
    print("\n------------------- DONE -------------------\n")
