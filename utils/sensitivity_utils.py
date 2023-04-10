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


def save_large(args, n_alpha, n_theta, KMS_mean, KMS_std, LSMC_mean, LSMC_std, ground_truth,
               time_KMS, time_LSMC):
    jnp.save(f"{args.save_path}/LSMC_mean_X_{n_alpha}_y_{n_theta}.npy", LSMC_mean)
    jnp.save(f"{args.save_path}/LSMC_std_X_{n_alpha}_y_{n_theta}.npy", LSMC_std)
    jnp.save(f"{args.save_path}/KMS_mean_X_{n_alpha}_y_{n_theta}.npy", KMS_mean)
    jnp.save(f"{args.save_path}/KMS_std_X_{n_alpha}_y_{n_theta}.npy", KMS_std)
    jnp.save(f"{args.save_path}/ground_truth.npy", ground_truth)

    time_dict = {}
    time_dict["BMC"] = None
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = None
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
