# Benchmark suite for PDX

# Helper: add a benchmark executable linked against PDX, with access to benchmark_utils.hpp
function(pdx_add_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE PDX)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${name} PRIVATE CMAKE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endfunction()

pdx_add_benchmark(BenchmarkPDXIVF pdx_ivf.cpp)
pdx_add_benchmark(BenchmarkEndToEnd pdx_end_to_end.cpp)
pdx_add_benchmark(BenchmarkSerialization pdx_serialization.cpp)
pdx_add_benchmark(BenchmarkFiltered pdx_filtered.cpp)
pdx_add_benchmark(BenchmarkSpecialFilters pdx_special_filtered.cpp)
pdx_add_benchmark(BenchmarkInsertion pdx_insertion.cpp)
pdx_add_benchmark(BenchmarkWorkload pdx_workload.cpp)
pdx_add_benchmark(BenchmarkIterativeFiltered pdx_iterative_filtered.cpp)

add_custom_target(benchmarks
    DEPENDS
        BenchmarkPDXIVF
        BenchmarkEndToEnd
        BenchmarkSerialization
        BenchmarkFiltered
        BenchmarkSpecialFilters
        BenchmarkInsertion
        BenchmarkWorkload
        BenchmarkIterativeFiltered
)
