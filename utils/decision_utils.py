import jax.numpy as jnp
import pickle
from scipy.stats import norm
import matplotlib.pyplot as plt


def save(args, Nx, Ny, rmse_BMC, rmse_KMS, rmse_LSMC, calibration_1, calibration_2):
    rmse_dict = {}
    rmse_dict["BMC"] = rmse_BMC
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    with open(f"{args.save_path}/rmse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    jnp.save(f"{args.save_path}/calibration_X_{Nx}_y_{Ny}", calibration_1)
    jnp.save(f"{args.save_path}/calibration_X_{Nx}_y_{Ny}", calibration_2)
    return


def scale(Z):
    s = Z.std()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std


def calibrate(ground_truth, BMC_mean, BMC_std):
    """
    Calibrate the BMC mean and std
    :param ground_truth: (N, )
    :param BMC_mean: (N, )
    :param BMC_std: (N, )
    :return:
    """
    BMC_std /= 10
    confidence_level = jnp.arange(0.0, 1.01, 0.05)
    prediction_interval = jnp.zeros(len(confidence_level))
    for i, c in enumerate(confidence_level):
        z_score = norm.ppf(1 - (1 - c) / 2)  # Two-tailed z-score for the given confidence level
        prob = jnp.less(jnp.abs(ground_truth - BMC_mean), z_score * BMC_std)
        prediction_interval = prediction_interval.at[i].set(prob.mean())

    plt.figure()
    plt.plot(confidence_level, prediction_interval, label="Model calibration", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration", color="black")
    plt.xlabel("Credible")
    plt.ylabel("Coverage")
    plt.title("Calibration plot")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.close()
    plt.show()
    return prediction_interval