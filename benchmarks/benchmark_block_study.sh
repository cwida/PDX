#!/bin/bash

n_vectors=(64 128 512 1024 4096 8192 16384 65536 131072)
dimensions=(8 16 32 64 128 192 256 384 512 768 1024 1536 2048 4096 8192)

make KernelNaryL2
make KernelPDXL2_16
make KernelPDXL2_32
make KernelPDXL2_64
make KernelPDXL2_128
make KernelPDXL2_256
make KernelPDXL2_512

for n_vector in "${n_vectors[@]}"
do
    for dimension in "${dimensions[@]}"
    do
        echo "Kernel Nary L2 | N=$n_vector D=$dimension"
        ./benchmarks/KernelNaryL2 ${n_vector} ${dimension}

        echo "Kernel PDX L2 16 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_16 $n_vector $dimension

        echo "Kernel PDX L2 32 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_32 $n_vector $dimension

        echo "Kernel PDX L2 64 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_64 $n_vector $dimension

        echo "Kernel PDX L2 128 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_128 $n_vector $dimension

        echo "Kernel PDX L2 256 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_256 $n_vector $dimension

        echo "Kernel PDX L2 512 | N=$n_vector D=$dimension"
        ./benchmarks/KernelPDXL2_512 $n_vector $dimension
    done
done