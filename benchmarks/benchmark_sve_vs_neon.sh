#!/bin/bash

n_vectors=(64 128 512 1024 4096 8192 16384 65536 131072)
dimensions=(8 16 32 64 128 192 256 384 512 768 1024 1536 2048 4096 8192)

make G4SVE
make G4NEON

for n_vector in "${n_vectors[@]}" 
do
    for dimension in "${dimensions[@]}"
    do
        echo "SVE L1 | N=$n_vector D=$dimension"
        ./benchmarks/G4SVE ${n_vector} ${dimension}

        echo "NEON L1 | N=$n_vector D=$dimension"
        ./benchmarks/G4NEON ${n_vector} ${dimension}

    done
done
