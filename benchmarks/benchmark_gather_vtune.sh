#!/bin/bash

n_vectors=(128 512 1024 4096 8192 16384 65536 131072)
dimensions=(16 32 64 128 192 256 384 512 768 1024 1536 2048)

n_vectors=(64)
dimensions=(8)

make KernelPDXL2
make KernelNaryL2
make PureScanGATHER

for n_vector in "${n_vectors[@]}"
do
    for dimension in "${dimensions[@]}"
    do
        echo "Kernel GATHER | N=$n_vector D=$dimension"
        vtune -collect memory-access -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/GATHER${n_vector}x${dimension} -- /home/ubuntu/data/PDX/benchmarks/PureScanGATHER ${n_vector} ${dimension}
        vtune -report hw-events -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/GATHER${n_vector}x${dimension} -report-output /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/GATHER${n_vector}x${dimension}.csv -format csv -csv-delimiter comma -- ./benchmarks/PureScanGATHER ${n_vector} ${dimension}

        echo "Kernel PDX | N=$n_vector D=$dimension"
        vtune -collect memory-access -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/PDX${n_vector}x${dimension} -- /home/ubuntu/data/PDX/benchmarks/KernelPDXL2 ${n_vector} ${dimension}
        vtune -report hw-events -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/PDX${n_vector}x${dimension} -report-output /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/PDX${n_vector}x${dimension}.csv -format csv -csv-delimiter comma -- ./benchmarks/KernelPDXL2 ${n_vector} ${dimension}

        echo "Kernel Nary L2 | N=$n_vector D=$dimension"
        vtune -collect memory-access -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/NARY${n_vector}x${dimension} -- /home/ubuntu/data/PDX/benchmarks/KernelNaryL2 ${n_vector} ${dimension}
        vtune -report hw-events -result-dir /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/NARY${n_vector}x${dimension} -report-output /home/ubuntu/data/PDX/benchmarks/results/SAPPHIRE/VTUNE/NARY${n_vector}x${dimension}.csv -format csv -csv-delimiter comma -- ./benchmarks/KernelNaryL2 ${n_vector} ${dimension}

    done
done