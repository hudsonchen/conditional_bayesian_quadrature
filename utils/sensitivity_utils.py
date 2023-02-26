import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd


def init_logging():
    save_dict = {}
    save_dict['True Value'] = []
    save_dict['BMC'] = []
    save_dict['MC'] = []
    return save_dict


def save_final_results(args, MC_list, cbq_mean_dict, cbq_std_dict, poly_mean_dict,
         IS_mean_dict, true_value, N_alpha_list, N_beta_list):
    jnp.save(f'{args.save_path}/MC', jnp.array(MC_list))

    pd.DataFrame(data=cbq_mean_dict).to_csv(f'{args.save_path}/BMC_mean.csv', encoding='utf-8', index=False)
    pd.DataFrame(data=cbq_std_dict).to_csv(f'{args.save_path}/BMC_std.csv', encoding='utf-8', index=False)
    pd.DataFrame(data=poly_mean_dict).to_csv(f'{args.save_path}/poly.csv', encoding='utf-8', index=False)
    pd.DataFrame(data=IS_mean_dict).to_csv(f'{args.save_path}/importance_sampling.csv', encoding='utf-8', index=False)


    fig, axs = plt.subplots(len(N_alpha_list), 1, figsize=(10, len(N_alpha_list) * 3))
    for i, ax in enumerate(axs):
        Nx = N_alpha_list[i]
        axs[i].set_ylim(true_value * 0.8, true_value * 1.2)
        axs[i].axhline(y=true_value, linestyle='--', color='black', label='true value')
        axs[i].plot(N_beta_list, MC_list, color='b', label='MC')
        axs[i].plot(N_beta_list, cbq_mean_dict[f"{Nx}"], color='r', label=f'CBQ Nx = {Nx}')
        axs[i].plot(N_beta_list, IS_mean_dict[f"{Nx}"], color='darkgreen', label=f'IS Nx = {Nx}')
        axs[i].plot(N_beta_list, poly_mean_dict[f"{Nx}"], color='brown', label=f'Poly Nx = {Nx}')
        axs[i].fill_between(N_beta_list, cbq_mean_dict[f"{Nx}"] - 2 * cbq_std_dict[f"{Nx}"],
                            cbq_mean_dict[f"{Nx}"] + 2 * cbq_std_dict[f"{Nx}"], color='r', alpha=0.5)
        axs[i].legend()
        # axs[i].set_xscale('log')
    # plt.tight_layout()
    plt.suptitle("Bayesian sensitivity analysis")
    plt.savefig(f"{args.save_path}/all_methods.pdf")
    plt.show()


def update_log(args, Nx, Ny, logging, true_value, MC, BMC):
    logging['True Value'].append(true_value)
    logging['BMC'].append(BMC)
    logging['MC'].append(MC)
    pd.DataFrame(data=logging).to_csv(f'{args.save_path}/logging__Nx_{Nx}__Ny_{Ny}.csv', encoding='utf-8', index=False)
    return logging


def scale(Z):
    s = Z.mean()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std
