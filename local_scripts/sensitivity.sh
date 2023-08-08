#!/bin/bash

# Loop to run the script 10 times
for i in {1..10}
do
  echo "Running iteration $i"
  cd /home/zongchen/fx_bayesian_quaduature/CBQ/
  python bayes_sensitivity.py --dim 2 --fn f3 --kernel_x rbf --kernel_theta matern --baseline_use_variance
done

echo "All iterations completed."
