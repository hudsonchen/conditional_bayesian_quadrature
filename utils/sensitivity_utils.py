import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from sobol_seq import i4_sobol_generate


def compute_rmse(ground_truth, CBQ_mean, BQ_mean, KMS_mean, LSMC_mean, IS_mean):
    rmse_CBQ = jnp.sqrt(jnp.mean((CBQ_mean - ground_truth) ** 2))
    rmse_BQ = jnp.sqrt(jnp.sqrt(jnp.mean((BQ_mean - ground_truth) ** 2)))
    rmse_KMS = jnp.sqrt(jnp.mean((KMS_mean - ground_truth) ** 2))
    rmse_LSMC = jnp.sqrt(jnp.mean((LSMC_mean - ground_truth) ** 2))
    rmse_IS = jnp.sqrt(jnp.mean((IS_mean - ground_truth) ** 2))
    return rmse_CBQ, rmse_BQ, rmse_KMS, rmse_LSMC, rmse_IS


def save(args, T, N, rmse_CBQ, rmse_BQ, rmse_KMS, rmse_LSMC, rmse_IS,
         time_CBQ, time_BQ, time_KMS, time_LSMC, time_IS, calibration):
    rmse_dict = {}
    rmse_dict["CBQ"] = rmse_CBQ
    rmse_dict["BQ"] = rmse_BQ
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    rmse_dict["IS"] = rmse_IS
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["CBQ"] = time_CBQ
    time_dict["BQ"] = time_BQ
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)

    jnp.save(f"{args.save_path}/calibration_T_{T}_N_{N}", calibration)
    return


def save_large(args, T, N, rmse_KMS, rmse_LSMC, rmse_IS, time_KMS, time_LSMC, time_IS):
    rmse_dict = {}
    rmse_dict["CBQ"] = None
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    rmse_dict["IS"] = rmse_IS
    with open(f"{args.save_path}/rmse_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["CBQ"] = None
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_T_{T}_N_{N}", 'wb') as f:
        pickle.dump(time_dict, f)
    return


def scale(Z):
    s = Z.mean()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std


def qmc_gaussian(mu, sigma, nsamples):
    """
    :param mu: (D, )
    :param sigma: (D, D)
    :param nsamples:
    :return: samples: (nsamples, D)
    """
    D = mu.shape[0]
    u = i4_sobol_generate(D, nsamples)
    L = jnp.linalg.cholesky(sigma)
    samples = mu[:, None] + (norm.ppf(u) @ L).T
    samples = samples.T
    return samples, u


def qmc_uniform(min_val, max_val, D, nsamples):
    """
    :param min_val:
    :param max_val:
    :param nsamples:
    :return:
    """
    u = i4_sobol_generate(D, nsamples)
    samples = min_val + u * (max_val - min_val)
    return samples


def calibrate(ground_truth, CBQ_mean, CBQ_std):
    """
    Calibrate the CBQ mean and std
    :param ground_truth: (N, )
    :param CBQ_mean: (N, )
    :param CBQ_std: (N, )
    :return:
    """
    CBQ_std /= 10
    confidence_level = jnp.arange(0.0, 1.01, 0.05)
    prediction_interval = jnp.zeros(len(confidence_level))
    for i, c in enumerate(confidence_level):
        z_score = norm.ppf(1 - (1 - c) / 2)  # Two-tailed z-score for the given confidence level
        prob = jnp.less(jnp.abs(ground_truth - CBQ_mean), z_score * CBQ_std)
        prediction_interval = prediction_interval.at[i].set(prob.mean())

    plt.figure()
    plt.plot(confidence_level, prediction_interval, label="Model calibration", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration", color="black")
    plt.xlabel("Confidence")
    plt.ylabel("Coverage")
    plt.title("Calibration plot")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.close()
    # plt.show()
    return prediction_interval
