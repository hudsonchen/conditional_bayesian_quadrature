import jax.numpy as jnp
import jax
import math
from tensorflow_probability.substrates import jax as tfp


def my_Matern(x, y, l):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :return: N*M
    """
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    K = kernel.matrix(x, y)
    return K


# @jax.jit
def dx_Matern(x, y, l):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :return: N*M*D
    """
    N, D = x.shape
    M = y.shape[0]
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_x_K_fn = jax.grad(kernel.apply, argnums=0)
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    return dx_K


# @jax.jit
def dy_Matern(x, y, l):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :return: N*M*D
    """
    N, D = x.shape
    M = y.shape[0]
    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_y_K_fn = jax.grad(kernel.apply, argnums=1)
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy).reshape(N, M, D)
    return dy_K


# @jax.jit
def dxdy_Matern(x, y, l):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :return: N*M
    """
    N, D = x.shape
    M = y.shape[0]

    kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=l)
    grad_xy_K_fn = jax.jacfwd(jax.jacrev(kernel.apply, argnums=1), argnums=0)

    def diag_sum_grad_xy_K_fn(x, y):
        return jnp.diag(grad_xy_K_fn(x, y)).sum()

    vec_grad_xy_K_fn = jax.vmap(diag_sum_grad_xy_K_fn, in_axes=(0, 0), out_axes=0)
    x_dummy = jnp.stack([x] * N, axis=1).reshape(N * M, D)
    y_dummy = jnp.stack([y] * M, axis=0).reshape(N * M, D)
    dxdy_K = vec_grad_xy_K_fn(x_dummy, y_dummy).reshape(N, M)
    return dxdy_K


# @jax.jit
def my_RBF(x, y, l):
    """
    :param x: N*D
    :param y: M*D
    :param l: scalar
    :return: N*M
    """
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=l)
    K = kernel.matrix(x, y)
    return K


def jax_dist(x, y):
    return jnp.sqrt(((x - y) ** 2).sum(-1)).squeeze()

distance = jax.vmap(jax_dist, in_axes=(None, 0), out_axes=1)
sign_func = jax.vmap(jnp.greater, in_axes=(None, 0), out_axes=1)


def my_Laplace(x, y, l):
    r = distance(x, y).squeeze()
    return jnp.exp(- r / l)


def dx_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-sign)
    return part1


def dy_Laplace(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * sign
    return part1


def dxdy_Laplace(x, y, l):
    r = distance(x, y).squeeze()
    part1 = jnp.exp(- r / l) * (-1)
    return part1


def one_d_my_Laplace(x, y, l):
    r = jax_dist(x, y).squeeze()
    return jnp.exp(- r / l)


def main():
    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(rng_key, shape=(3, 2))
    rng_key, _ = jax.random.split(rng_key)
    y = jax.random.uniform(rng_key, shape=(3, 2))
    l = 0.5
    batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    K1 = batch_kernel.matrix(x, y)
    K2 = my_Matern(x, y, 0.5)
    print(K1)
    print(K2)

    print("============")
    batch_kernel = tfp.math.psd_kernels.MaternThreeHalves(amplitude=1., length_scale=.5)
    grad_x_K_fn = jax.grad(batch_kernel.apply, argnums=(0,))
    vec_grad_x_K_fn = jax.vmap(grad_x_K_fn, in_axes=(0, 0), out_axes=1)
    grad_y_K_fn = jax.grad(batch_kernel.apply, argnums=(1,))
    vec_grad_y_K_fn = jax.vmap(grad_y_K_fn, in_axes=(0, 0), out_axes=1)

    seed = 0
    rng_key = jax.random.PRNGKey(seed)
    N = 2
    D = 3
    l = 0.5

    rng_key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(rng_key, shape=(N, D))
    rng_key, _ = jax.random.split(rng_key)
    y = jax.random.uniform(rng_key, shape=(N, D))

    x_dummy = jnp.stack((x, x), axis=0).reshape(N * N, D)
    y_dummy = jnp.stack((y, y), axis=1).reshape(N * N, D)

    dx_K = vec_grad_x_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)
    dy_K = vec_grad_y_K_fn(x_dummy, y_dummy)[0].reshape(N, N, D)

    print(dx_K)
    print(dy_K)

    print(grad_x_K_fn(x[0, :], y[0, :]))
    print(grad_x_K_fn(x[0, :], y[1, :]))
    print(grad_x_K_fn(x[1, :], y[0, :]))
    print(grad_x_K_fn(x[1, :], y[1, :]))

    print(grad_y_K_fn(x[0, :], y[0, :]))
    print(grad_y_K_fn(x[0, :], y[1, :]))
    print(grad_y_K_fn(x[1, :], y[0, :]))
    print(grad_y_K_fn(x[1, :], y[1, :]))

if __name__ == '__main__':
    main()