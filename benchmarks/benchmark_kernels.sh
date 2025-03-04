#!/bin/bash

n_vectors=(64 128 512 1024 4096 8192 16384 65536 131072)
dimensions=(8 16 32 64 128 192 256 384 512 768 1024 1536 2048 4096 8192)

make KernelPDXL1
make KernelNaryL1
make KernelPDXL2
make KernelNaryL2
make KernelPDXIP
make KernelNaryIP

for n_vector in "${n_vectors[@]}" 
do
    for dimension in "${dimensions[@]}"
    do
        echo "Kernel PDX L1 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL1 ${n_vector} ${dimension}

        echo "Kernel Nary L1 | N=$n_vector D=$dimension"
        ./benchmarks/KernelNaryL1 ${n_vector} ${dimension}

        echo "Kernel PDX L2 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2 ${n_vector} ${dimension}

        echo "Kernel Nary L2 | N=$n_vector D=$dimension"
        ./benchmarks/KernelNaryL2 ${n_vector} ${dimension}

        echo "Kernel PDX IP | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXIP ${n_vector} ${dimension}

        echo "Kernel Nary IP | N=$n_vector D=$dimension"
        ./benchmarks/KernelNaryIP ${n_vector} ${dimension}
    done
done
