import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import jax
import numpy as np
import pandas as pd

eps = 1e-6


def convert_dict_to_jnp(D):
    D_copy = {}
    for k in D.keys():
        D_copy[k] = jnp.array(D[k])
    return D_copy


def non_zero_ind(A):
    ind = jnp.abs(A) > 2.0
    return ind


def scale(A):
    m = 0.
    s = jnp.log10(A.max())
    std = 10 ** (int(s))
    return m, std, (A - m) / std