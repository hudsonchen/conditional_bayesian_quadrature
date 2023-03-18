## This line is for requesting the memory
#$ -l tmem=10G
## This line is for requesting the gpu
#$ -l gpu=true
#$ -pe gpu 1
## This line is for the running time
#$ -l h_rt=5:0:0
## I don't know what it does, maybe priority?
#$ -R y

#$ -S /bin/bash
#$ -j y
## This line is the name of your job
#$ -N cbq_toy

## This line is activating python and the conda environment
#source /share/apps/source_files/python/python-3.8.5.source
conda activate cbq

## Some basic commands to show the job is running
date
nvidia-smi

## Check if the environment is correct.
which pip
which python

pwd
python /home/zongchen/CBQ/stein_toy_cbq.py