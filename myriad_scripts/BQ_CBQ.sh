#$ -l mem=10G
#$ -l h_rt=6:0:0
#$ -R y
#$ -S /bin/bash
#$ -wd /home/ucabzc9/Scratch/
#$ -j y
#$ -N bq_cbq

conda activate cbq

date

## Check if the environment is correct.
which pip
which python

pwd
python /home/zongchen/CBQ/BQ_CBQ.py