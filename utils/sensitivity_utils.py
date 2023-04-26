import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm


def save(args, n_alpha, n_theta, mse_BMC, mse_KMS, mse_LSMC, mse_IS,
         time_BMC, time_KMS, time_LSMC, time_IS, calibration):
    mse_dict = {}
    mse_dict["BMC"] = mse_BMC
    mse_dict["KMS"] = mse_KMS
    mse_dict["LSMC"] = mse_LSMC
    mse_dict["IS"] = mse_IS
    with open(f"{args.save_path}/mse_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
        pickle.dump(mse_dict, f)

    time_dict = {}
    time_dict["BMC"] = time_BMC
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
        pickle.dump(time_dict, f)

    jnp.save(f"{args.save_path}/calibration_X_{n_alpha}_y_{n_theta}", calibration)
    return


def save_large(args, n_alpha, n_theta, mse_KMS, mse_LSMC, mse_IS, time_KMS, time_LSMC, time_IS):
    mse_dict = {}
    mse_dict["BMC"] = None
    mse_dict["KMS"] = mse_KMS
    mse_dict["LSMC"] = mse_LSMC
    mse_dict["IS"] = mse_IS
    with open(f"{args.save_path}/mse_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
        pickle.dump(mse_dict, f)

    time_dict = {}
    time_dict["BMC"] = None
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
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
    plt.xlabel("Confidence")
    plt.ylabel("Coverage")
    plt.title("Calibration plot")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.close()
    # plt.show()
    return prediction_interval
