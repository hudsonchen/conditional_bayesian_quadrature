# Conditional Bayesian Quadrature

This repository contains the implementation of the code for the paper "Conditional Bayesian Quadrature".

## Installation

To install the necessary requirements, use the following command:

pip install -r requirements.txt

## Reproducing Results

### Bayesian Sensitivity Analysis

To reproduce the results for Bayesian sensitivity analysis, run the following command:

python sensitivity_conjugate.py --dim 2 --g_fn g3 --kernel_x RBF  --kernel_theta Matern

### Black-Scholes Model

To reproduce the results for the Black-Scholes model using Stein kernels, run:

python finance.py --kernel_theta rbf --kernel_x stein_matern

And to reproduce the results for the Black-Scholes model not using Stein kernels, run:

python finance.py --kernel_theta rbf --kernel_x log_normal_rbf

### SIR Model

To reproduce the results for the SIR model, run:

python SIR.py

### Health Economics

To reproduce the results for health economics, run:

python decision.py
