#!/bin/bash

n_vectors=(64 128 512 1024 4096 8192 16384 65536 131072 262144)
dimensions=(8 16 32 64 128 192 256 384 512 768 1024 1536 2048 4096 8192)

make PureScanGATHER
make KernelPDXL2
make KernelNaryL2

for n_vector in "${n_vectors[@]}" 
do
    for dimension in "${dimensions[@]}"
    do
         echo "Kernel GATHER | N=$n_vector D=$dimension"
         ./benchmarks/PureScanGATHER ${n_vector} ${dimension}

        echo "Kernel PDX | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2 ${n_vector} ${dimension}

        echo "Kernel SIMD | N=$n_vector D=$dimension"
        ./benchmarks/KernelNaryL2 ${n_vector} ${dimension}

    done
done
