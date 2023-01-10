import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy import integrate
from functools import partial
from tqdm import tqdm

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()


def Geometric_Brownian(n, dt, sigma=0.3, S0=1):
    dWt = np.random.normal(0, np.sqrt(dt), size=(int(n),))
    dlnSt = sigma * dWt - 0.5 * (sigma ** 2) * dt
    St = np.exp(np.cumsum(dlnSt) + np.log(S0))
    return St


def callBS(t, s, K, sigma):
    part_one = np.log(s / K) / (sigma * np.sqrt(t)) + 0.5 * sigma * np.sqrt(t)
    part_two = np.log(s / K) / (sigma * np.sqrt(t)) - 0.5 * sigma * np.sqrt(t)
    return s * norm.cdf(part_one) - K * norm.cdf(part_two)


def BSM_butterfly_analytic():
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50

    # St = Geometric_Brownian(100, t / 100, sigma=sigma, S0=S0)[-1]

    epsilon = np.random.normal(0, 1)
    St = S0 * np.exp(sigma * np.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
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


def GP(X, Y, gY, g, x_prime, kernel_x, kernel_y, compute_phi='empirical', eps = 1e-3):
    Nx = X.shape[0]
    Ny = Y.shape[1]

    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import Matern

    if kernel_x == 'rbf':
        Kx = RBF(length_scale=0.3)
    else:
        raise NotImplementedError

    if kernel_y == 'rbf':
        Ky = RBF(length_scale=0.1)
    elif kernel_y == 'matern':
        Ky = 1.0 * Matern(length_scale=0.3, nu=3.5)
    elif kernel_y == 'exponential':  ## Currently the best one for piece-wise function
        def exponential(X1, X2, length_scale):
            dists = cdist(X1 / length_scale, X2 / length_scale, metric="euclidean")
            K = np.exp(- dists / length_scale)
            return K
        Ky = partial(exponential, length_scale=1.)
    else:
        raise NotImplementedError

    for i in range(Nx):
        Yi = Y[i, :][:, None]
        Yi_mean = Yi.mean()
        Yi_std = Yi.std()
        Yi_std = 1.0 if Yi_std == 0.0 else Yi_std   # To prevent numerical issue
        Yi_standardized = (Yi - Yi_mean) / Yi_std

        gYi = gY[i, :][:, None]
        gYi_mean = gYi.mean()
        gYi_std = gYi.std()
        gYi_std = 1.0 if gYi_std == 0.0 else gYi_std   # To prevent numerical issue
        gYi_standardized = (gYi - gYi_mean) / gYi_std
        # \mu = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'
        y_prime = np.linspace(-3, 3, 100)[:, None]

        Ky_inv = np.linalg.inv(Ky(Yi_standardized, Yi_standardized) + eps * np.eye(Ny))
        K_y_prime_y = Ky(y_prime, Yi_standardized)
        mu_standardized = K_y_prime_y @ Ky_inv @ gYi_standardized
        std_standardized = np.sqrt(np.diag(Ky(y_prime, y_prime) - K_y_prime_y @ Ky_inv @ K_y_prime_y.T))[:, None]
        mu_original = mu_standardized * gYi_std + gYi_mean
        std_original = std_standardized * gYi_std

        plt.figure()
        y_prime_original = y_prime * Yi_std + Yi_mean
        plt.plot(y_prime_original.squeeze(), mu_original.squeeze())
        plt.fill_between(y_prime_original.squeeze(), (mu_original - std_original).squeeze(),
                         (mu_original + std_original).squeeze(), alpha=0.5)
        plt.scatter(Y, gY)
        plt.title(f"This is {kernel_y} kernel")
        plt.show()
        pause = True
        return


def cbq(X, Y, gY, g, x_prime, sigma, kernel_x, kernel_y, compute_phi='empirical', eps=1e-3):
    """
    :param X: X is of size Nx
    :param Y: Y is of size Nx * Ny
    :param gY: gY is g(Y)
    :param g: g is a function, that takes Y as input and return value g(Y).
    :param x_prime: is the target conditioning value of x, should be of shape [1, 1]
    :param compute_phi: choose the mode to compute kernel mean embedding phi.
    :return: return the expectation E[g(Y)|X=x_prime]
    """

    Nx = X.shape[0]
    Ny = Y.shape[1]

    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import Matern

    if kernel_x == 'rbf': # This is the best kernel for x
        Kx = RBF(length_scale=1.0)
    elif kernel_x == 'matern':
        Kx = 1.0 * Matern(length_scale=0.3, nu=3.5)
    elif kernel_x == 'exponential':  ## Currently the best one for piece-wise function
        def exponential(X1, X2, length_scale):
            dists = cdist(X1 / length_scale, X2 / length_scale, metric="euclidean")
            K = np.exp(- dists / length_scale)
            return K
        Kx = partial(exponential, length_scale=1.)
    else:
        raise NotImplementedError

    if kernel_y == 'rbf':
        Ky = RBF(length_scale=0.1)
    elif kernel_y == 'matern':
        Ky = 1.0 * Matern(length_scale=0.3, nu=3.5)
    elif kernel_y == 'exponential':  # Currently the best one for piece-wise function
        def exponential(X1, X2, length_scale):
            dists = cdist(X1 / length_scale, X2 / length_scale, metric="euclidean")
            K = np.exp(- dists / length_scale)
            return K
        Ky = partial(exponential, length_scale=1.)
    else:
        raise NotImplementedError

    mu_list = []
    std_list = []
    for i in range(Nx):
        Yi = Y[i, :][:, None]
        Yi_mean = Yi.mean()
        Yi_std = Yi.std()
        Yi_std = 1.0 if Yi_std == 0.0 else Yi_std   # To prevent numerical issue
        Yi_standardized = (Yi - Yi_mean) / Yi_std

        gYi = gY[i, :][:, None]
        gYi_mean = gYi.mean()
        gYi_std = gYi.std()
        gYi_std = 1.0 if gYi_std == 0.0 else gYi_std   # To prevent numerical issue
        gYi_standardized = (gYi - gYi_mean) / gYi_std
        # phi = \int ky(Y, y)p(y|x)dy, varphi = \int \int ky(y', y)p(y|x)p(y|x)dydy'
        if compute_phi == 'empirical':
            phi = Ky(Yi_standardized, Yi_standardized).mean(1)
            varphi = Ky(Yi_standardized, Yi_standardized).mean()
        elif compute_phi == 'numerical_integral':
            # It is the convolution of a normal rv and a log-normal rv.
            # p(Y=y|X=x) = \frac{x^2}{y} \frac{1}{\sqrt{2\pi} \sigma} \exp{-\frac{1}{2 \sigma^2} (\log{y/x} + \sigma^2 / x)^2}
            # Note that we have standardization, so take mean and std into account.
            def log_normal(y, x):
                y = y * Yi_std + Yi_mean
                # part1 = (x ** 2 / y) * (1. / (np.sqrt(2 * math.pi) * sigma))
                # part2 = np.exp(-0.5 / (sigma ** 2) * (np.log(y / x) + sigma ** 2 / 2) ** 2)
                normal_samples = (np.log(y) - np.log(x) + 0.5 * (sigma ** 2)) / sigma
                density = norm.pdf(normal_samples)
                return density

            def normal_log_normal(y, yi, x):
                y = np.array([[y]])
                return Ky(y, yi) * log_normal(y, x)

            x = X[i]
            phi = np.array([])
            for j in range(Ny):
                yi = Yi_standardized[j, :][:, None]
                integrand = partial(normal_log_normal, yi=yi, x=x)
                value, _ = integrate.quad(integrand, -2, 2)
                phi = np.append(phi, value)
            phi = phi[:, None]
            varphi = Ky(Yi_standardized, Yi_standardized).mean()
        else:
            raise NotImplementedError

        Ky_inv = np.linalg.inv(Ky(Yi_standardized, Yi_standardized) + eps * np.eye(Ny))
        mu_standardized = phi.T @ Ky_inv @ gYi_standardized
        std_standardized = np.sqrt(varphi - phi.T @ Ky_inv @ phi)
        mu_original = mu_standardized * gYi_std + gYi_mean
        std_original = std_standardized * gYi_std
        mu_list.append(mu_original)
        std_list.append(std_original)

        # Large sample mu
        # price(X[i], 10000)[1].mean()

    Sigma = np.diag(np.array(std_list))
    Mu = np.array(mu_list)

    mean_Mu = Mu.mean()
    std_Mu = Mu.std()
    std_Mu = 1.0 if std_Mu == 0.0 else std_Mu  # To prevent numerical issue
    Mu_standardized = (Mu - mean_Mu) / std_Mu
    Sigma_standardized = Sigma / std_Mu
    mean_X = X.mean()
    std_X = X.std()
    std_X = 1.0 if std_X == 0.0 else std_X  # To prevent numerical issue
    X_standardized = (X - mean_X) / std_X
    x_prime_standardized = (x_prime - mean_X) / std_X

    Kx_inv = np.linalg.inv(Kx(X_standardized, X_standardized) + Sigma_standardized)
    mu_y_x_prime = Kx(x_prime_standardized, X_standardized) @ Kx_inv @ Mu_standardized
    var_y_x_prime = Kx(x_prime_standardized, x_prime_standardized) - Kx(x_prime_standardized, X_standardized) @ Kx_inv @ Kx(X_standardized, x_prime_standardized)
    std_y_x_prime = np.sqrt(var_y_x_prime)

    mu_y_x_prime_original = mu_y_x_prime * std_Mu + mean_Mu
    std_y_x_prime_original = std_y_x_prime * std_Mu

    # The is for choosing the best kernel for x.
    # x_test = np.linspace(20, 120, 100)[:, None]
    # x_test_standardized = (x_test - mean_X) / std_X
    # K_X_x_test = Kx(X_standardized, x_test_standardized)
    # mu_y_x_test = K_X_x_test.T @ Kx_inv @ Mu_standardized
    # var_y_x_test = Kx(x_test_standardized, x_test_standardized) - K_X_x_test.T @ Kx_inv @ K_X_x_test
    # std_y_x_test = np.sqrt(np.diag(var_y_x_test))
    # mu_y_x_test_original = mu_y_x_test * std_Mu + mean_Mu
    # std_y_x_test_original = std_y_x_test * std_Mu
    #
    # true_X = np.load('./finance_X.npy')
    # true_EgY_X = np.load('./finance_EgY_X.npy')
    #
    # plt.figure()
    # plt.scatter(X.squeeze(), gY.mean(1))
    # plt.plot(x_test.squeeze(), mu_y_x_test_original.squeeze(), label='predict')
    # plt.plot(true_X, true_EgY_X, label='true')
    # plt.fill_between(x_test.squeeze(), mu_y_x_test_original.squeeze() - std_y_x_test_original,
    #                  mu_y_x_test_original.squeeze() + std_y_x_test_original, alpha=0.5)
    # plt.plot()
    # plt.legend()
    # plt.show()
    pause = True
    return mu_y_x_prime_original, std_y_x_prime_original


def price(St, N, K1=50, K2=150, s=-0.2, sigma=0.3, T=2, t=1, visualize=False):
    """
    :param St: the price St at time t
    :return: The function returns the price ST at time T sampled from the conditional
    distribution p(ST|St), and the loss \psi(ST) - \psi((1+s)ST) due to the shock. Their shape is Nx * Ny
    """
    output_shape = (St.shape[0], N)
    epsilon = np.random.normal(0, 1, size=output_shape)
    ST = St * np.exp(sigma * np.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
    psi_ST_1 = np.maximum(ST - K1, 0) + np.maximum(ST - K2, 0) - 2 * np.maximum(ST - (K1 + K2) / 2, 0)
    psi_ST_2 = np.maximum((1 + s) * ST - K1, 0) + np.maximum((1 + s) * ST - K2, 0) - 2 * np.maximum(
        (1 + s) * ST - (K1 + K2) / 2, 0)

    if visualize:
        loss_list = []
        ST_list = []
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs = axs.flatten()

        St_dummy = St[0]
        log_ST_min = np.log(St_dummy) + sigma * np.sqrt((T - t)) * (-3) - 0.5 * (sigma ** 2) * (T - t)
        log_ST_max = np.log(St_dummy) + sigma * np.sqrt((T - t)) * (+3) - 0.5 * (sigma ** 2) * (T - t)
        ST_min = np.exp(log_ST_min)
        ST_max = np.exp(log_ST_max)
        ST_samples = np.linspace(ST_min, ST_max, 100)
        normal_samples = (np.log(ST_samples) - np.log(St_dummy) + 0.5 * (sigma ** 2) * (T - t)) / sigma
        density = norm.pdf(normal_samples)
        axs[0].plot(ST_samples, density)
        axs[0].set_title(r"The pdf for $p(S_T|S_t)$")
        axs[0].set_xlabel(r"$S_T$")

        for _ in range(1000):
            epsilon = np.random.normal(0, 1, size=(1, 1))
            ST = St_dummy * np.exp(sigma * np.sqrt((T - t)) * epsilon - 0.5 * (sigma ** 2) * (T - t))
            psi_ST_1 = np.maximum(ST - K1, 0) + np.maximum(ST - K2, 0) - 2 * np.maximum(ST - (K1 + K2) / 2, 0)
            psi_ST_2 = np.maximum((1 + s) * ST - K1, 0) + np.maximum((1 + s) * ST - K2, 0) - 2 * np.maximum(
                (1 + s) * ST - (K1 + K2) / 2, 0)
            loss_list.append(psi_ST_1 - psi_ST_2)
            ST_list.append(ST)

        ST_dummy = np.array(ST_list).squeeze()
        loss_dummy = np.array(loss_list).squeeze()
        axs[1].scatter(ST_dummy, loss_dummy)
        axs[1].set_title(r"$\psi(S_T) - \psi((1+s)S_T)$")
        axs[1].set_xlabel(r"$S_T$")
        plt.suptitle(rf"$S_t$ is {St_dummy[0]}")
        plt.show()
    return ST, psi_ST_1 - psi_ST_2


def save_true_value():
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50
    epsilon = np.random.normal(0, 1, size=(1000, 1))
    St = S0 * np.exp(sigma * np.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
    _, loss = price(St, 100000, K1=K1, K2=K2, s=s, sigma=sigma, T=T, t=t)
    St = St.squeeze()
    ind = np.argsort(St)
    value = loss.mean(1)
    np.save('./finance_X.npy', St[ind])
    np.save('./finance_EgY_X.npy', value[ind])
    plt.figure()
    plt.plot(St[ind], value[ind])
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\mathbb{E}[g(Y) \mid X]$")
    plt.title("True value for finance experiment")
    plt.show()
    plt.savefig("./true_distribution.pdf")
    return


def cbq_option_pricing(visualize=False):
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50
    Nx_array = np.array([1, 3, 5, 10, 20, 100])
    Ny_array = np.arange(1, 100, 2)
    BMC_dict = {}
    MC_list = []

    St_prime = np.array([[50.0]])
    if visualize:
        price(St_prime, N=1, visualize=True)


    # True value with standard MC
    for _ in range(1):
        true_value = price(St_prime, 1000000)[1].mean()
        print('True Value is:', true_value)

    for Nx in Nx_array:
        BMC_array = np.array([])
        for Ny in tqdm(Ny_array):
            epsilon = np.random.normal(0, 1, size=(Nx, 1))
            St = S0 * np.exp(sigma * np.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
            ST, loss = price(St, Ny, K1=K1, K2=K2, s=s, sigma=sigma, T=T, t=t)

            # This is use standard GP to fit loss at a given St, mainly used for choosing kernel on Y.
            # GP(St, ST, loss, price, St_prime, kernel_x='rbf', kernel_y='exponential')

            mu_y_x_prime, std_y_x_prime = cbq(St, ST, loss, price, St_prime, sigma=sigma,
                                              kernel_x='rbf', kernel_y='exponential')
            BMC_array = np.append(BMC_array, mu_y_x_prime[0][0])
        BMC_dict[f"{Nx}"] = BMC_array

    for Ny in Ny_array:
        MC_list.append(price(St_prime, Ny)[1].mean())

    fig, axs = plt.subplots(len(Nx_array), 1, figsize=(10, len(Nx_array) * 3))
    for i, ax in enumerate(axs):
        Nx = Nx_array[i]
        axs[i].set_ylim(1, 7)
        axs[i].axhline(y=true_value, linestyle='--', color='black', label='true value')
        axs[i].plot(Ny_array, MC_list, color='b', label='MC')
        axs[i].plot(Ny_array, BMC_dict[f"{Nx}"], color='r', label=f'BMC Nx = {Nx}')
        axs[i].legend()
        # axs[i].set_xscale('log')
    plt.show()
    return


def main():
    visualize_brownian = False
    debug_BSM = False
    debug_cbq = True
    if visualize_brownian:
        n = 100.
        T = 1.
        dt = T / n
        plt.figure()
        for i in range(10):
            St = Geometric_Brownian(n, dt)
            plt.plot(St)
        plt.show()
    elif debug_BSM:
        BSM_butterfly_analytic()
    elif debug_cbq:
        cbq_option_pricing(visualize=True)
    else:
        pass
    return


if __name__ == '__main__':
    main()
    # save_true_value()
#
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
