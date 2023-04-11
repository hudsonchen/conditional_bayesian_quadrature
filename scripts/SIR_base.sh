#$ -l tmem=20G
#$ -l h_vmem=20G
#$ -l h_rt=12:0:0
#$ -R y

#$ -S /bin/bash
#$ -j y
#$ -N cbq_SIR

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
mpirun -np 1 python /home/zongchen/CBQ/SIR.py $JOB_PARAMS