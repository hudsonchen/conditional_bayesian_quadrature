import jax.numpy as jnp


def polynomial(X, Y, gY, X_prime, poly=4):
    """
    Polynomial Regression
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    X_poly = jnp.ones_like(X)
    for i in range(1, poly + 1):
        X_poly = jnp.concatenate((X_poly, X ** i), axis=1)
    eps = 1e-6
    D = (1 + poly) * X.shape[1]
    theta = jnp.linalg.inv(X_poly.T @ X_poly + eps * jnp.eye(D)) @ X_poly.T @ gY.mean(1)

    X_prime_poly = jnp.ones_like(X_prime)
    for i in range(1, poly + 1):
        X_prime_poly = jnp.concatenate((X_prime_poly, X_prime ** i), axis=1)
    phi = X_prime_poly @ theta
    std = 0
    pause = True
    return phi, std


def importance_sampling(py_x_fn, X, Y, gY, X_prime):
    """
    Self-normalized importance sampling
    :param py_x_fn:
    :param X_prime: N_test*D
    :param X: Nx*D
    :param Y: Nx*Ny*D
    :param gY: Nx*Ny
    :return:
    """
    Nx, Ny = Y.shape[0], Y.shape[1]
    N_test = X_prime.shape[0]
    IS_prime_list = []
    for j in range(N_test):
        IS_list = []
        x_prime = X_prime[j, :][:, None]
        for i in range(Nx):
            xi = X[i, :][:, None]
            Yi = Y[i, :, :]
            gYi = gY[i, :][:, None]
            py_x_prime = py_x_fn(theta=Yi, alpha=x_prime)
            py_x_i = py_x_fn(theta=Yi, alpha=xi)
            weight = py_x_prime / py_x_i
            mu = (weight * gYi).mean()
            IS_list.append(mu)
        IS_prime_list.append(jnp.array(IS_list).mean())
        pause = True
    return jnp.array(IS_prime_list), 0
