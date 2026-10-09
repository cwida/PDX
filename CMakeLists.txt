cmake_minimum_required(VERSION 3.26)
project(PDX)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
set(PDX_PORTABLE OFF CACHE BOOL "Use portable SIMD flags instead of -march=native (for wheel builds)")
set(PDX_MARCH "native" CACHE STRING "Target architecture for -march (e.g. native, haswell, znver4). Empty string disables -march.")
set(PDX_SKIP_FFTW OFF CACHE BOOL "Skip FFTW dependency entirely")
set(PDX_COMPILE_TESTS OFF CACHE BOOL "Whether to compile tests")
set(PDX_COMPILE_BENCHMARKS OFF CACHE BOOL "Whether to compile benchmarks")
set(PDX_COMPILE_PYTHON ${PROJECT_IS_TOP_LEVEL} CACHE BOOL "Whether to compile the Python bindings (when pybind11 is found)")

# ── Compiler flags (top-level only) ──
if(PROJECT_IS_TOP_LEVEL)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    endif()
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")

include(FetchContent)
include(CTest)

# ── SuperKMeans ──
# Index construction. Its target also brings Eigen, FFTW (optional), the executor and GEMM backends
# (SKMEANS_EXECUTOR, SKMEANS_GEMM) and the architecture flags (-march, MSVC, Emscripten).
set(SKMEANS_PORTABLE ${PDX_PORTABLE} CACHE BOOL "" FORCE)
set(SKMEANS_MARCH "${PDX_MARCH}" CACHE STRING "" FORCE)
set(SKMEANS_SKIP_FFTW ${PDX_SKIP_FFTW} CACHE BOOL "" FORCE)
set(SKMEANS_COMPILE_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_Declare(superkmeans
    GIT_REPOSITORY https://github.com/lkuffo/SuperKMeans.git
    GIT_TAG a3f6b1048a9c297705156c3873073fb7b52eaa50
)
FetchContent_MakeAvailable(superkmeans)

# ── INTERFACE library target ──
add_library(PDX INTERFACE)
add_library(PDX::PDX ALIAS PDX)

# Suppress warnings from our headers when used as a dependency
if(NOT PROJECT_IS_TOP_LEVEL)
    set(_PDX_SYSTEM SYSTEM)
endif()

target_include_directories(PDX ${_PDX_SYSTEM} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PDX INTERFACE superkmeans::superkmeans)
target_compile_features(PDX INTERFACE cxx_std_17)

# ── Subdirectories ──
if(PDX_COMPILE_TESTS)
    enable_testing()
    message(STATUS "Tests enabled")
    add_subdirectory(tests)
else()
    message(STATUS "Tests disabled")
endif()

if(PDX_COMPILE_BENCHMARKS)
    message(STATUS "Benchmarks enabled")
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmarks disabled")
endif()

# ── Python bindings ──
if(PDX_COMPILE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module QUIET)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG QUIET)
endif()

if(PDX_COMPILE_PYTHON AND Python_FOUND AND pybind11_FOUND)
    message(STATUS "Python bindings enabled")
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
    message(STATUS "pybind11 found: ${pybind11_VERSION}")

    pybind11_add_module(compiled python/lib.cpp)
    target_link_libraries(compiled PRIVATE PDX)
    if(NOT MSVC)
        target_compile_options(compiled PRIVATE -Wall -Wno-unknown-pragmas)
    endif()

    install(TARGETS compiled
        LIBRARY DESTINATION pdxearch
        COMPONENT python
    )
else()
    message(STATUS "Python bindings disabled")
endif()
