#$ -l tmem=10G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l h_rt=12:0:0
#$ -R y

#$ -S /bin/bash
#$ -j y
#$ -N cbq_sensitivity

#source /share/apps/source_files/python/python-3.8.5.source

JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

conda activate cbq

date
nvidia-smi

## Check if the environment is correct.
which pip
which python

pwd
python /home/zongchen/CBQ/sensitivity.py $JOB_PARAMS