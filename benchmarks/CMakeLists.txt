include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(BENCH_COMMON_LIBS ${MKL_COMMON_LIBS} ${BLAS_LINK_LIBRARIES})

if (FFTW_FOUND)
    message(STATUS "Linking FFTW: ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB}")
    list(APPEND BENCH_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

add_executable(BenchmarkPDXIVF pdx_ivf.cpp)
add_executable(BenchmarkEndToEnd pdx_end_to_end.cpp)
add_executable(BenchmarkSerialization pdx_serialization.cpp)
add_executable(BenchmarkFiltered pdx_filtered.cpp)
add_executable(BenchmarkSpecialFilters pdx_special_filtered.cpp)

target_link_libraries(BenchmarkPDXIVF ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkEndToEnd ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkSerialization ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkFiltered ${BENCH_COMMON_LIBS})
target_link_libraries(BenchmarkSpecialFilters ${BENCH_COMMON_LIBS})

add_custom_target(benchmarks
    DEPENDS
        BenchmarkPDXIVF
        BenchmarkEndToEnd
        BenchmarkSerialization
        BenchmarkFiltered
        BenchmarkSpecialFilters
)
