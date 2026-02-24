cmake_minimum_required(VERSION 3.26)
project(PDX)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")

add_compile_options(-fPIC)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/SuperKMeans/include)

find_package(OpenMP REQUIRED)

list(PREPEND CMAKE_PREFIX_PATH /usr/local)

set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG QUIET)
if (MKL_FOUND)
    message(STATUS "MKL library found")
    message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
    get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "MKL includes: ${mkl_includes}")
    add_definitions(-DEIGEN_USE_MKL_ALL)
    set(MKL_COMMON_LIBS MKL::MKL OpenMP::OpenMP_CXX m dl)
    set(BLAS_LINK_LIBRARIES "")
else()
    set(MKL_COMMON_LIBS "")
    message(STATUS "MKL not found. Trying to find a BLAS implementation")

    # On macOS, prefer Apple's Accelerate framework over other BLAS implementations
    if(APPLE)
        set(BLA_VENDOR Apple)
        message(STATUS "macOS detected: prioritizing Apple Accelerate framework")
    endif()

    find_package(BLAS REQUIRED)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    add_definitions(-DEIGEN_USE_BLAS)
    set(BLAS_LINK_LIBRARIES ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "extern/findFFTW")
#find_package(FFTW QUIET)
#if (FFTW_FLOAT_LIB_FOUND)
#    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIRS})
#    add_definitions(-DHAS_FFTW)
#    include_directories(${FFTW_INCLUDE_DIRS})
#else()
#    message(WARNING "FFTW (+float capabilities) not found, we recommend you install it")
#endif()

add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include extern/Eigen)
add_subdirectory(benchmarks)
