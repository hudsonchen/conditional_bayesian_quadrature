# Conditional Bayesian Quadrature

This code contains an anonymous implementation of the code for the paper **_Conditioal Bayesian Quadrature_**;

## Installation

To install requirements:
```setup
$ pip install -r requirements.txt
```

## Reproducing results
To reproduce the results for Bayesian sensitivity analysis
```python
python sensitivity_conjugate.py --dim 2 --g_fn g3 --kernel_x RBF  --kernel_theta Matern
```

To reproduce the results for Black-Scholes\
using stein kernels
```python
python finance.py --kernel_theta rbf --kernel_x stein_matern
```

not using stein kernels
```python
python finance.py --kernel_theta rbf --kernel_x log_normal_rbf
```

To reproduce the results for SIR model
```python
python SIR.py
```

To reproduce the results for health economics 
```python
python decision.py
```