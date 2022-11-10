#!/bin/bash -l

# Batch script to run an OpenMP threaded job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:10:0

# Request 1 gigabyte of RAM for each core/thread
# (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N Multi-threaded Job

# Request 16 cores.
#$ -pe smp 16

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/<your_UCL_id>/Scratch/output

# 8. Run the application.
python
pwd

