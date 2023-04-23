#$ -l tmem=80G
#$ -l h_vmem=80G
#$ -l h_rt=15:0:0
#$ -pe smp 1

#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N cbq_finance

#source /share/apps/source_files/python/python-3.8.5.source
JOB_PARAMS=$(sed "${SGE_TASK_ID}q;d" "$1")
echo "Job params: $JOB_PARAMS"

conda activate cbq

date

## Check if the environment is correct.
which pip
which python

pwd
module load openmpi
mpirun -np 1 python /home/zongchen/CBQ/finance.py $JOB_PARAMS