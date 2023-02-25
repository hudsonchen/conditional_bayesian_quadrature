echo "$1"
PARAMS=$1
echo $PARAMS

SGE_TASK_ID=1
JOB=$(sed "${SGE_TASK_ID}q;d" "$1")
echo $JOB

SGE_TASK_ID=2
JOB=$(sed "${SGE_TASK_ID}q;d" "$1")
echo $JOB

jobs_in_parallel=$(wc -l < "$1")
echo $jobs_in_parallel

