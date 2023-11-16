#!/bin/bash

#num_proc = (8 16 32 64)

#num_input = (65536 262144 1048576 4194304 16777216 67108864 268435456)

for num_proc in 2 4 8 16 32 64; do
    for size in 65536 262144 1048576 4194304 16777216 67108864; do
        sbatch mpi.grace_job "$size" "$num_proc"
        echo "ran for process $num_proc at size $size"
    done
done