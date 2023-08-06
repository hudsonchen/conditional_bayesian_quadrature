#!/bin/bash

# Loop to run the script 10 times
# for i in {1..10}
# do
#   echo "Running iteration $i"
#   cd /home/zongchen/fx_bayesian_quaduature/CBQ/
#   python black_scholes.py --kernel_x log_normal_rbf --kernel_theta rbf --baseline_use_variance
# done

for i in {1..10}
do
  echo "Running iteration $i"
  cd /home/zongchen/fx_bayesian_quaduature/CBQ/
  python black_scholes.py --kernel_x log_normal_rbf --kernel_theta rbf --baseline_use_variance
done

# for i in {1..10}
# do
#   echo "Running iteration $i"
#   cd /home/zongchen/fx_bayesian_quaduature/CBQ/
#   python black_scholes.py --kernel_x stein_matern --kernel_theta rbf
# done

# for i in {1..10}
# do
#   echo "Running iteration $i"
#   cd /home/zongchen/fx_bayesian_quaduature/CBQ/
#   python black_scholes.py --kernel_x stein_matern --kernel_theta rbf

echo "All iterations completed."
