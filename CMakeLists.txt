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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")

# ── CMake modules ──
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# ── Accumulator variables for the INTERFACE target ──
set(_PDX_LINK_LIBS "")
set(_PDX_COMPILE_DEFS "")
set(_PDX_INCLUDE_DIRS "")

# ── OpenMP ──
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    if(APPLE)
        message(FATAL_ERROR
            "OpenMP not found. On macOS, install it with: brew install libomp\n"
            "Then re-run CMake.")
    else()
        message(FATAL_ERROR "OpenMP not found.")
    endif()
endif()
list(APPEND _PDX_LINK_LIBS OpenMP::OpenMP_CXX)

# ── BLAS / MKL ──
list(PREPEND CMAKE_PREFIX_PATH /usr/local)

set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG QUIET)
if(MKL_FOUND)
    message(STATUS "MKL library found")
    message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
    get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "MKL includes: ${mkl_includes}")
    list(APPEND _PDX_LINK_LIBS MKL::MKL m dl)
    list(APPEND _PDX_COMPILE_DEFS EIGEN_USE_MKL_ALL)
else()
    message(STATUS "MKL not found. Trying to find a BLAS implementation")

    # On macOS, prefer Apple's Accelerate framework over other BLAS implementations
    if(APPLE)
        set(BLA_VENDOR Apple)
        message(STATUS "macOS detected: prioritizing Apple Accelerate framework")
    endif()

    find_package(BLAS QUIET)
    if(NOT BLAS_FOUND)
        if(APPLE)
            message(FATAL_ERROR "BLAS not found. Apple Accelerate should always be available on macOS.")
        else()
            message(STATUS "System BLAS not found. Building OpenBLAS from source via FetchContent...")
            FetchContent_Declare(openblas
                URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31.tar.gz
            )
            set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "" FORCE)
            set(C_LAPACK OFF CACHE BOOL "" FORCE)
            set(USE_OPENMP ON CACHE BOOL "" FORCE)
            set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)
            set(NUM_THREADS 384 CACHE STRING "" FORCE)
            set(NOFORTRAN ON CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(openblas)
            list(APPEND _PDX_LINK_LIBS openblas)
        endif()
    else()
        message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
        list(APPEND _PDX_LINK_LIBS ${BLAS_LIBRARIES})
    endif()
endif()

# ── FFTW (optional) ──
if(NOT PDX_SKIP_FFTW)
    find_package(FFTW QUIET)
    if(FFTW_FLOAT_LIB_FOUND)
        message(STATUS "FFTW (+float capabilities) found: ${FFTW_INCLUDE_DIRS}")
        list(APPEND _PDX_COMPILE_DEFS HAS_FFTW)
        list(APPEND _PDX_LINK_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
        list(APPEND _PDX_INCLUDE_DIRS ${FFTW_INCLUDE_DIRS})
    else()
        message(STATUS "FFTW (+float capabilities) not found, we recommend you install them for increased performance")
    endif()
else()
    message(STATUS "FFTW skipped (PDX_SKIP_FFTW=ON)")
endif()

# ── Eigen ──
# Use a consumer-provided Eigen3::Eigen target when available to avoid leaking our
# bundled headers into the consumer's include path. Otherwise, use bundled extern/Eigen.
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen: using existing Eigen3::Eigen target")
    list(APPEND _PDX_LINK_LIBS Eigen3::Eigen)
else()
    message(STATUS "Eigen: using bundled extern/Eigen")
    # Exposed as Eigen3::Eigen so SuperKMeans below shares this copy instead of its own
    add_library(pdx_bundled_eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS pdx_bundled_eigen)
    target_include_directories(pdx_bundled_eigen INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen)
    list(APPEND _PDX_LINK_LIBS Eigen3::Eigen)
endif()

# ── SuperKMeans (index construction; brings cpuinfo, numkong and ruy through its own CMake) ──
set(SKMEANS_PORTABLE ${PDX_PORTABLE} CACHE BOOL "" FORCE)
set(SKMEANS_MARCH "${PDX_MARCH}" CACHE STRING "" FORCE)
# FFTW is found, linked and HAS_FFTW-defined by PDX above; FindFFTW cannot run a second time
set(SKMEANS_SKIP_FFTW ON CACHE BOOL "" FORCE)
add_subdirectory(extern/SuperKMeans)
list(APPEND _PDX_LINK_LIBS superkmeans::superkmeans)

# ── INTERFACE library target ──
add_library(PDX INTERFACE)
add_library(PDX::PDX ALIAS PDX)

# Suppress warnings from our headers when used as a dependency
if(NOT PROJECT_IS_TOP_LEVEL)
    set(_PDX_SYSTEM SYSTEM)
endif()

target_include_directories(PDX ${_PDX_SYSTEM} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_PDX_INCLUDE_DIRS}
)

target_link_libraries(PDX INTERFACE ${_PDX_LINK_LIBS})
target_compile_definitions(PDX INTERFACE ${_PDX_COMPILE_DEFS})
target_compile_features(PDX INTERFACE cxx_std_17)

# Architecture flags on the target (propagated to consumers)
if(PDX_PORTABLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        target_compile_options(PDX INTERFACE $<$<CONFIG:Release>:-O3 -mavx2 -mfma>)
        message(STATUS "Portable mode: x86_64, using -mavx2 -mfma")
    else()
        target_compile_options(PDX INTERFACE $<$<CONFIG:Release>:-O3>)
        message(STATUS "Portable mode: ${CMAKE_SYSTEM_PROCESSOR}, using generic -O3")
    endif()
elseif(PDX_MARCH)
    target_compile_options(PDX INTERFACE $<$<CONFIG:Release>:-O3 -march=${PDX_MARCH}>)
    message(STATUS "Architecture: -march=${PDX_MARCH}")
else()
    target_compile_options(PDX INTERFACE $<$<CONFIG:Release>:-O3>)
    message(STATUS "Architecture: no -march flag")
endif()

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
    target_compile_options(compiled PRIVATE -Wall -Wno-unknown-pragmas)

    install(TARGETS compiled
        LIBRARY DESTINATION pdxearch
        COMPONENT python
    )
else()
    message(STATUS "Python bindings disabled")
endif()
