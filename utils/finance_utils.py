import jax
import jax.numpy as jnp
import time
from jax.scipy.stats import norm


def Geometric_Brownian(n, dt, rng_key, sigma=0.3, S0=1):
    rng_key, _ = jax.random.split(rng_key)
    dWt = jax.random.normal(rng_key, shape=(int(n),)) * jnp.sqrt(dt)
    dlnSt = sigma * dWt - 0.5 * (sigma ** 2) * dt
    St = jnp.exp(jnp.cumsum(dlnSt) + jnp.log(S0))
    return St


def callBS(t, s, K, sigma):
    part_one = jnp.log(s / K) / (sigma * jnp.sqrt(t)) + 0.5 * sigma * jnp.sqrt(t)
    part_two = jnp.log(s / K) / (sigma * jnp.sqrt(t)) - 0.5 * sigma * jnp.sqrt(t)
    return s * norm.cdf(part_one) - K * norm.cdf(part_two)


def BSM_butterfly_analytic():
    seed = int(time.time())
    # seed = 0
    rng_key = jax.random.PRNGKey(seed)
    rng_key, _ = jax.random.split(rng_key)
    K1 = 50
    K2 = 150
    s = -0.2
    t = 1
    T = 2
    sigma = 0.3
    S0 = 50

    # St = Geometric_Brownian(100, t / 100, sigma=sigma, S0=S0)[-1]

    epsilon = jax.random.normal(rng_key)
    St = S0 * jnp.exp(sigma * jnp.sqrt(t) * epsilon - 0.5 * (sigma ** 2) * t)
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

def scale(Z):
    scale = Z.mean()
    standardized = Z / scale
    return standardized, scale


def standardize(Z):
    mean = Z.mean()
    std = Z.std()
    standardized = (Z - mean) / std
    return standardized, mean, std
