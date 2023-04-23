## Request memory. I think it will request this much memory per core (as set by smp below), so in
## this case 20GB.
#$ -l mem=20G
## Max allowed run time. Set it as low as possible to get scheduled sooner.
#$ -l h_rt=1:00:00
#$ -pe smp 4
## I don't completely understand this, but I think it tries to start reserving cores as they become
## available, rather than waiting for 4 to be available at once, to avoid single core jobs jumping
## ahead of this job.
#$ -R y
## Say we want bash to sh.
#$ -S /bin/bash
## Write both stderr and stdout to a log file.
#$ -wd /home/ucabzc9/Scratch/
## Can't remember what this does.
#$ -j y
## This is an array job, and we want to run elements 1,2,3,4 of the array.
#$ -t 1
## The name of the job.
#$ -N test

## This fetches which element of the array this job is.
#number=$SGE_TASK_ID
## These two lines read the configuration I want from a text file, reading the appropriate
## configuration for the array element.
#paramfile=/home/ucabotk/Scratch/ilbp_output/fine_tuning_to_load_cifar100.txt
#load_run="`sed -n ${number}p $paramfile`"
#

# Running date and nvidia-smi is useful to get some info in case the job crashes.

#module -f unload compilers mpi gcc-libs
#module load beta-modules
#module load gcc-libs/10.2.0
#module load cuda/11.2.0/gnu-10.2.0

## Load conda
module -f unload compilers
module load compilers/gnu/4.9.2
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate /lustre/home/ucabzc9/.conda/envs/cbq

## Print out the date and nivida-smi for debugging
date

## Check if the environment is correct.
which pip
which python3

pwd
gerun python3 /home/ucabzc9/Scratch/CBQ/SIR.py --mode peak_number