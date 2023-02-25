#!/bin/bash

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel

qsub -t 1-${jobs_in_parallel} /home/zongchen/CBQ/scripts/sensitivity_base.sh "$1"
