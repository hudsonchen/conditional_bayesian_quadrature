## First we generate the samples

import math
import numpy as np
import matplotlib.pyplot as plt
import copy

np.random.seed(1)
M = 100
eps = 1e-6

t = np.random.uniform(0, 2 * math.pi, size=[M, 1])
n1 = np.random.normal(0, 0.1, size=[M, 1])
n2 = np.random.normal(0, 0.1, size=[M, 1])
x = np.sin(t) + n1
y = np.cos(t) + n2

# plt.figure()
# plt.scatter(x, y, color='b')
# plt.show()

from sklearn.gaussian_process.kernels import RBF

Gaussian_kernel = RBF(length_scale=0.1) + eps * np.eye(M)

Kx = Gaussian_kernel(x, x)

j = 0
R = np.zeros([M, M])
d = copy.deepcopy(np.diag(Kx))
nu = np.zeros(M)
I = np.zeros(M, dtype=int)
a, I[j] = np.max(d), np.argmax(d)
eta = 1e-2
while a > eta:
    nu[j] = np.sqrt(a)
    for i in range(M):
        R[j, i] = (Kx[int(I[j]), i] - R[:, i].T @ R[:, int(I[j])]) / nu[j]
        d[i] = d[i] - R[j, i] ** 2
    j = j + 1
    a, I[j] = np.max(d), np.argmax(d)


T = j
RR = R[:T, :]
c = 0