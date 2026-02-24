include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(BENCH_COMMON_LIBS ${MKL_COMMON_LIBS} ${BLAS_LINK_LIBRARIES})

if (FFTW_FOUND)
    message(STATUS "Linking FFTW: ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB}")
    list(APPEND BENCH_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

add_executable(BenchmarkPDXIVF pdx_ivf.cpp)
add_executable(BenchmarkEndToEnd pdx_end_to_end.cpp)
add_executable(BenchmarkSerialization pdx_serialization.cpp)

target_link_libraries(BenchmarkPDXIVF ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkEndToEnd ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkSerialization ${BENCH_COMMON_LIBS})

# Filtered benchmarks (need rewriting for new interface)
# add_executable(FilteredBenchmarkU8IVFTreeADSampling pdx_ivf2_adsampling_u8_filtered.cpp)
# add_executable(FilteredBenchmarkPDXADSampling pdx_ivf_adsampling_filtered.cpp)
# target_link_libraries(FilteredBenchmarkPDXADSampling ${BENCH_COMMON_LIBS})
# target_link_libraries(FilteredBenchmarkU8IVFTreeADSampling ${BENCH_COMMON_LIBS})
