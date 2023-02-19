import numpy as np
from scipy.integrate import odeint
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import pwd
import os
import pickle
import scipy

if pwd.getpwuid(os.getuid())[0] == 'hudsonchen':
    os.chdir("/Users/hudsonchen/research/fx_bayesian_quaduature/CBQ")
elif pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir("/home/zongchen/CBQ")
else:
    pass


# The ODE
def dv_dt(v, t, alpha_tilde, beta_tilde, delta_tilde, gamma_tilde):
    v1 = v[0]
    v2 = v[1]
    dv1_dt = alpha_tilde * v1 - beta_tilde * v1 * v2
    dv2_dt = delta_tilde * v1 * v2 - gamma_tilde * v2
    return np.array([dv1_dt, dv2_dt])


# initial conditions
t_num = 100
Nx = 20
Ny = 3
eps = 1e-3
t = np.linspace(0, 10, t_num)


# parameters
def init_params():
    alpha = np.random.normal(0, 0.5)
    beta = np.random.normal(-3, 0.5)
    delta = np.random.normal(-3, 0.5)
    gamma = np.random.normal(0, 0.5)
    v1_0 = np.random.normal(np.log(10), 1)
    v2_0 = np.random.normal(np.log(10), 1)
    sigma_1 = 1
    sigma_2 = 1
    return alpha, beta, delta, gamma, sigma_1, sigma_2, v1_0, v2_0


# The integrand
def f(y):
    return y[:, :, 0] ** 2 + y[:, :, 1]


# Generate samples
for i in tqdm(range(Nx)):
    # if os.path.exists(f'./data/lotka/data_{i}_Ny_{Ny}'):
    # a = os.listdir(f'./data/lotka/')[0]
    # Ny_current = float(a.split('Ny_')[1])
    # if Ny_current == Ny:
    #     continue
    # else:
    #     pass
    # else:
    alpha, beta, delta, gamma, sigma_1, sigma_2, v1_0, v2_0 = init_params()
    param_args = {'alpha_tilde': np.exp(alpha), 'beta_tilde': np.exp(beta),
                  'delta_tilde': np.exp(delta), 'gamma_tilde': np.exp(gamma)}
    dv_dt_2 = partial(dv_dt, **param_args)
    v0 = np.array([np.exp(v1_0), np.exp(v2_0)])
    # solving ODE
    v = odeint(partial(dv_dt, **param_args), v0, t)
    param_args.update({'v1_0_tilde': np.exp(v1_0), 'v2_0_tilde': np.exp(v2_0)})
    param_args.update({'v': v})

    noise = np.random.normal([0, 0], [sigma_1, sigma_2], size=(Ny, t_num, 2))
    y = v + noise
    param_args.update({f'y_samples': y})
    param_args.update({f'fy_samples': f(y)})

    # Use many samples MC to obtain the true mean
    noise = np.random.normal([0, 0], [sigma_1, sigma_2], size=(10000, t_num, 2))
    y = v + noise
    fy_true = f(y).mean(0)
    param_args.update({f'fy_true': fy_true})
    pickle.dump(param_args, open(f'./data/lotka/data_{i}_Ny_{Ny}', "wb"))

X = np.zeros((Nx, 6))
Y = np.zeros((Nx, Ny, t_num, 2))
fY = np.zeros((Nx, Ny, t_num))
fy_true = np.zeros((Nx, t_num))

for i in range(Nx):
    all_dict = pickle.load(open(f'./data/lotka/data_{i}_Ny_{Ny}', "rb"))
    param_args = {'alpha_tilde': all_dict['alpha_tilde'], 'beta_tilde': all_dict['beta_tilde'],
                  'delta_tilde': all_dict['delta_tilde'], 'gamma_tilde': all_dict['gamma_tilde'],
                  'v1_0_tilde': all_dict['v1_0_tilde'], 'v2_0_tilde': all_dict['v2_0_tilde']}
    X[i, :] = np.array(list(param_args.values()))
    Y[i, :, :] = all_dict[f'y_samples']
    fY[i, :, :] = all_dict[f'fy_samples']
    fy_true[i, :] = all_dict[f'fy_true']


def kernel_x(x1, x2, l):
    r = cdist(x1, x2, 'euclidean')
    return np.exp(- r ** 2 / 2 / (l ** 2))


def kernel_y(y1, y2, l):
    r = cdist(y1, y2, 'euclidean')
    return np.exp(- r ** 2 / 2 / (l ** 2))


x_ind = 4
MC_array = 0 * fy_true[x_ind, :]
BMC_mean_array = 0 * fy_true[x_ind, :]
BMC_std_array = 0 * fy_true[x_ind, :]
ly = 1.0


def normalize(Y):
    mean = Y.mean(0)
    std = Y.std(0)
    return mean, std, (Y - mean) / std


for time in tqdm(range(t_num)):
    y_train = Y[x_ind, :, time, :]
    fy_train = fY[x_ind, :, time]
    fy_train = fy_train[:, None]

    y_train_mean, y_train_std, y_train_normalized = normalize(y_train)
    MC_array[time] = fy_train.mean()  # * fy_train_std + fy_train_mean

    Ky_inv = np.linalg.inv(kernel_y(y_train_normalized, y_train_normalized, ly) + eps * np.eye(Ny))
    phi = kernel_y(y_train_normalized, y_train_normalized, ly).mean(0)
    BMC_mean = phi @ Ky_inv @ fy_train
    BMC_mean_array[time] = BMC_mean  # * fy_train_std + fy_train_mean
    BMC_var = kernel_y(y_train, y_train, ly).mean() - phi @ Ky_inv @ phi.T
    BMC_std_array[time] = np.sqrt(BMC_var)  # * fy_train_std

plt.figure()
plt.plot(t, MC_array, color='r', label='MC')
plt.plot(t, BMC_mean_array, color='b', label='BMC')
plt.fill_between(t, BMC_mean_array - BMC_std_array, BMC_mean_array + BMC_std_array, color='b', alpha=0.5)
# plt.plot(t, fy_true[x_ind, :], color='black', label='True')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Y")
plt.show()

# alpha, beta, delta, gamma, sigma_1, sigma_2 = init_params()
# param_args = {'alpha_tilde': np.exp(alpha), 'beta_tilde': np.exp(beta),
#               'delta_tilde': np.exp(delta), 'gamma_tilde': np.exp(gamma)}
# x_test = np.array(list(param_args.values()))[None, :]
# dv_dt_2 = partial(dv_dt, **param_args)
# v = odeint(partial(dv_dt, **param_args), v0, t)
# y_test_true = v[50, 0]
# Y_train = Y.mean(1)[:, 50, 0]
# lx = 0.3
# K_x_inv = np.linalg.inv(kernel_x(X, X, lx) + eps * np.eye(Nx))
# y_test_pred = kernel_x(x_test, X, lx) @ K_x_inv @ Y_train
#
# print('True value is', y_test_true)
# print('Prediction is', y_test_pred)

# plt.figure()
# plt.plot(t, v[:, 0], color='r')
# plt.xlabel("Time")
# plt.ylabel("Y")
# plt.show()
#
# plt.figure()
# plt.plot(t, y[:, 0], color='b')
# plt.xlabel("Time")
# plt.ylabel("Y")
# plt.show()

pause = True
