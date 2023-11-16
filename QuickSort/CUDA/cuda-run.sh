#!/bin/bash

#num_proc = (8 16 32 64)

#num_input = (65536 262144 1048576 4194304 16777216 67108864 268435456)

for num_proc in 64 128 256 512; do
    sed -i "s/#define MAX_THREADS[[:space:]]*[0-9]\+/#define MAX_THREADS ${num_proc}/" quicksort.cu
    for size in 65536 262144 1048576 4194304 16777216 67108864; do
    
        sed -i "s/#define N[[:space:]]*[0-9]\+/#define N ${size}/" quicksort.cu
        make

        sbatch bitonic.grace_job "$size" "$num_proc"
        echo "ran for process $num_proc at size $size"
    done

done
