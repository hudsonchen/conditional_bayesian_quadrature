import jax.numpy as jnp
import jax
import math


def jax_dist(x, y):
    return jnp.abs(x - y).squeeze()

distance = jax.vmap(jax_dist, in_axes=(None, 1), out_axes=1)
sign_func = jax.vmap(jnp.greater, in_axes=(None, 1), out_axes=1)


# @jax.jit
def my_Matern(x, y, l):
    r = distance(x, y).squeeze()
    part1 = 1 + math.sqrt(3) * r / l
    part2 = jnp.exp(-math.sqrt(3) * r / l)
    return part1 * part2


# @jax.jit
def one_d_my_Matern(x, y, l):
    r = jax_dist(x, y).squeeze()
    part1 = 1 + math.sqrt(3) * r / l
    part2 = jnp.exp(-math.sqrt(3) * r / l)
    return part1 * part2


# @jax.jit
def dx_Matern(x, y, l):
    sign = sign_func(x, y).squeeze().astype(float) * 2 - 1
    r = distance(x, y).squeeze()
    part1 = jnp.exp(-math.sqrt(3) / l * r) * (math.sqrt(3) / l * sign)
    part2 = (-math.sqrt(3) / l * sign) * jnp.exp(-math.sqrt(3) / l * r) * (1 + math.sqrt(3) / l * r)
    return part1 + part2


# @jax.jit
def dy_Matern(x, y, l):
    sign = -(sign_func(x, y).squeeze().astype(float) * 2 - 1)
    r = distance(x, y).squeeze()
    part1 = jnp.exp(-math.sqrt(3) / l * r) * (math.sqrt(3) / l * sign)
    part2 = (-math.sqrt(3) / l * sign) * jnp.exp(-math.sqrt(3) / l * r) * (1 + math.sqrt(3) / l * r)
    return part1 + part2


# @jax.jit
def my_RBF(x, y, l):
    r = distance(x, y).squeeze()
    return jnp.exp(- r ** 2 / 2 / (l ** 2))


def my_laplace_kernel(x, y, l):
    r = distance(x, y).squeeze()
    return jnp.exp(- r / l)


def one_d_my_laplace_kernel(x, y, l):
    r = jax_dist(x, y).squeeze()
    return jnp.exp(- r / l)


# @jax.jit
def one_d_my_RBF(x, y, l):
    r = jax_dist(x, y).squeeze()
    return jnp.exp(- r ** 2 / 2 / (l ** 2))
