#!/bin/bash

# Loop to run the script 10 times
for i in {1..10}
do
  echo "Running iteration $i"
  cd /home/zongchen/fx_bayesian_quaduature/CBQ/
  python finance.py --dim 2 --g_fn g3 --kernel_x RBF --kernel_theta Matern --baseline_use_variance
done

for i in {1..10}
do
  echo "Running iteration $i"
  cd /home/zongchen/fx_bayesian_quaduature/CBQ/
  python finance.py --dim 2 --g_fn g3 --kernel_x RBF --kernel_theta Matern
done

echo "All iterations completed."
