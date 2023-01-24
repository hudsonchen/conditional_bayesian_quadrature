import jax
import jax.numpy as jnp


def scale(Z):
    scale = Z.mean()
    standardized = Z / scale
    return standardized, scale


def standardize(Z):
    mean = Z.mean()
    std = Z.std()
    standardized = (Z - mean) / std
    return standardized, mean, std
