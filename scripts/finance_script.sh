#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=12:0:0
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N cbq_finance

#source /share/apps/source_files/python/python-3.8.5.source
conda activate cbq

date

## Check if the environment is correct.
which pip
which python

pwd
python /home/zongchen/CBQ/finance.py --kernel_x rbf --kernel_y stein_matern