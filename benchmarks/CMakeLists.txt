# ADSampling
add_executable(BenchmarkNaryIVFADSampling bench_adsampling/nary_ivf_adsampling.cpp)
add_executable(BenchmarkNaryIVFADSamplingSIMD bench_adsampling/nary_ivf_adsampling_simd.cpp)
add_executable(BenchmarkPDXADSampling bench_adsampling/pdx_ivf_adsampling.cpp)

# BSA
add_executable(BenchmarkNaryIVFBSASIMD bench_bsa/nary_ivf_bsa_simd.cpp)
add_executable(BenchmarkPDXBSA bench_bsa/pdx_ivf_bsa.cpp)

# BOND
add_executable(BenchmarkPDXIVFBOND ./bench_bond/pdx_bond_ivf.cpp)
add_executable(BenchmarkPDXBOND ./bench_bond/pdx_bond.cpp)

# Linear
add_executable(BenchmarkPDXLinearScan ./bench_linear/pdx_linear_scan.cpp)
add_executable(BenchmarkNaryIVFLinearScan bench_linear/nary_ivf_linear_scan.cpp)

# Kernels experiment
add_executable(KernelPDXL1 ./bench_kernels/pdx_l1.cpp)
add_executable(KernelPDXL2 ./bench_kernels/pdx_l2.cpp)
add_executable(KernelPDXIP ./bench_kernels/pdx_ip.cpp)
add_executable(KernelNaryL1 ./bench_kernels/nary_l1.cpp)
add_executable(KernelNaryL2 ./bench_kernels/nary_l2.cpp)
add_executable(KernelNaryIP ./bench_kernels/nary_ip.cpp)


