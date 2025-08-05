# ADSampling
add_executable(BenchmarkPDXADSampling bench_adsampling/pdx_ivf_adsampling.cpp)
add_executable(BenchmarkIMIADSampling bench_adsampling/pdx_imi_adsampling.cpp)
add_executable(BenchmarkU8PDXADSampling bench_adsampling/pdx_ivf_adsampling_u8.cpp)
add_executable(BenchmarkU8IMIADSampling bench_adsampling/pdx_imi_adsampling_u8.cpp)

# BOND
add_executable(BenchmarkPDXIVFBOND ./bench_bond/pdx_bond_ivf.cpp)
add_executable(BenchmarkPDXBOND ./bench_bond/pdx_bond.cpp)

if (FFTW_FOUND)
    target_link_libraries(BenchmarkPDXADSampling ${FFTW_FLOAT_LIB})
    target_link_libraries(BenchmarkIMIADSampling ${FFTW_FLOAT_LIB})
    target_link_libraries(BenchmarkU8PDXADSampling ${FFTW_FLOAT_LIB})
    target_link_libraries(BenchmarkU8IMIADSampling ${FFTW_FLOAT_LIB})
    target_link_libraries(BenchmarkPDXIVFBOND ${FFTW_FLOAT_LIB})
    target_link_libraries(BenchmarkPDXBOND ${FFTW_FLOAT_LIB})
endif()
