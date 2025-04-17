# ADSampling
add_executable(BenchmarkNaryIVFADSampling bench_adsampling/nary_ivf_adsampling.cpp)
add_executable(BenchmarkNaryIVFADSamplingSIMD bench_adsampling/nary_ivf_adsampling_simd.cpp)
add_executable(BenchmarkPDXADSampling bench_adsampling/pdx_ivf_adsampling.cpp)
add_executable(BenchmarkU8PDXADSampling bench_adsampling/pdx_ivf_adsampling_u8.cpp)
add_executable(BenchmarkU6PDXADSampling bench_adsampling/pdx_ivf_adsampling_u6.cpp)
add_executable(BenchmarkU4PDXADSampling bench_adsampling/pdx_ivf_adsampling_u4.cpp)

# BSA
add_executable(BenchmarkNaryIVFBSASIMD bench_bsa/nary_ivf_bsa_simd.cpp)
add_executable(BenchmarkPDXBSA bench_bsa/pdx_ivf_bsa.cpp)

# BOND
add_executable(BenchmarkPDXIVFBOND ./bench_bond/pdx_bond_ivf.cpp)
add_executable(BenchmarkU8PDXIVFBOND ./bench_bond/pdx_bond_ivf_u8.cpp)
add_executable(BenchmarkPDXBOND ./bench_bond/pdx_bond.cpp)

# Linear
add_executable(BenchmarkPDXLinearScan ./bench_linear/pdx_linear_scan.cpp)
add_executable(BenchmarkU8PDXLinearScan ./bench_linear/pdx_linear_scan_u8.cpp)
add_executable(BenchmarkNaryIVFLinearScan bench_linear/nary_ivf_linear_scan.cpp)
add_executable(BenchmarkU8NaryLinearScan bench_linear/nary_linear_scan_u8.cpp)

# SVE vs nEON
add_executable(G4SVE ./bench_kernels/g4_sve.cpp)
add_executable(G4NEON ./bench_kernels/g4_neon.cpp)

# Kernels experiment
add_executable(KernelPDXL1 ./bench_kernels/pdx_l1.cpp)
add_executable(KernelPDXL2 ./bench_kernels/pdx_l2.cpp)
add_executable(KernelPDXIP ./bench_kernels/pdx_ip.cpp)
add_executable(KernelNaryL1 ./bench_kernels/nary_l1.cpp)
add_executable(KernelNaryL2 ./bench_kernels/nary_l2.cpp)
add_executable(KernelNaryIP ./bench_kernels/nary_ip.cpp)


