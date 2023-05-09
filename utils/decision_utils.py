import jax.numpy as jnp
import pickle


def save(args, n_alpha, n_theta, rmse_BMC, rmse_KMS, rmse_LSMC, rmse_IS,
         time_BMC, time_KMS, time_LSMC, time_IS, calibration):
    rmse_dict = {}
    rmse_dict["BMC"] = rmse_BMC
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    rmse_dict["IS"] = rmse_IS
    with open(f"{args.save_path}/rmse_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
        pickle.dump(rmse_dict, f)

    time_dict = {}
    time_dict["BMC"] = time_BMC
    time_dict["KMS"] = time_KMS
    time_dict["LSMC"] = time_LSMC
    time_dict["IS"] = time_IS
    with open(f"{args.save_path}/time_dict_X_{n_alpha}_y_{n_theta}", 'wb') as f:
        pickle.dump(time_dict, f)

    jnp.save(f"{args.save_path}/calibration_X_{n_alpha}_y_{n_theta}", calibration)
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
