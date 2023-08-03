#!/bin/bash
source activate cbq

# Loop to run the script 10 times
for i in {1..10}
do
  echo "Running iteration $i"
  cd /home/zongchen/fx_bayesian_quaduature/CBQ/
  python sensitivity_conjugate.py --dim 2 --g_fn g3 --kernel_x RBF --kernel_theta Matern --baseline_use_variance
done

echo "All iterations completed."