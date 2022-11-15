## Request memory. I think it will request this much memory per core (as set by smp below), so in
## this case 20GB.
#$ -l mem=5G
## Max allowed run time. Set it as low as possible to get scheduled sooner.
#$ -l h_rt=1:00:00
## Request 1 GPU.
#$ -l gpu=1
## Only run on nodes with specific GPUs. I had to do this because some nodes had newer gpus that
## weren't supported by the cuda version available. They might have fixed this though. You can look
## up the node types on the myriad help pages.
#$ -ac allow=JEF
## Request 4 cores. It's important to request multiple cores in order to not bottleneck the GPU.
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
#$ -t 1-4
## The name of the job.
#$ -N test

## This fetches which element of the array this job is.
#number=$SGE_TASK_ID
## These two lines read the configuration I want from a text file, reading the appropriate
## configuration for the array element.
#paramfile=/home/ucabotk/Scratch/ilbp_output/fine_tuning_to_load_cifar100.txt
#load_run="`sed -n ${number}p $paramfile`"
#
## You might have to play with these to match the pytorch version you are using.
#module load cuda/10.1.243
#module load python3/3.9

# Running date and nvidia-smi is useful to get some info in case the job crashes.
date
nvidia-smi
#python3 $HOME/interlocking-backprop/examples/run_fine_tuning.py --load_run $load_run --dataset cifar100
