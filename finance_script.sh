#$ -l tmem=10G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y

#$ -S /bin/bash
#$ -j y
#$ -N cbq_finance

source /share/apps/source_files/python/python-3.8.5.source
conda activate cbq

date
nvidia-smi

## Check if the environment is correct.
which pip
which python3

pwd
python3 /home/zongchen/CBQ/finance.py --kernel_x rbf --kernel_y stein_matern