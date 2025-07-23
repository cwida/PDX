# ADSampling
add_executable(BenchmarkPDXADSampling bench_adsampling/pdx_ivf_adsampling.cpp)
add_executable(BenchmarkIMIADSampling bench_adsampling/pdx_imi_adsampling.cpp)
add_executable(BenchmarkU8PDXADSampling bench_adsampling/pdx_ivf_adsampling_u8.cpp)
add_executable(BenchmarkU8IMIADSampling bench_adsampling/pdx_imi_adsampling_u8.cpp)

# BOND
add_executable(BenchmarkPDXIVFBOND ./bench_bond/pdx_bond_ivf.cpp)
add_executable(BenchmarkPDXBOND ./bench_bond/pdx_bond.cpp)



