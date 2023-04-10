import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle


def save(args, n_alpha, n_theta, mse_BMC, mse_KMS, mse_LSMC, mse_IS,
                                   time_BMC, time_KMS, time_LSMC, time_IS):
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
