cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
project(PDX)

add_compile_options(-fPIC)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)
# include(ExternalProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "extern/findFFTW")
find_package(FFTW QUIET)
if (FFTW_FLOAT_LIB_FOUND)
    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIR})
    add_definitions(-DHAS_FFTW)
    include_directories(${FFTW_INCLUDE_DIR})
else()
    message(WARNING "FFTW (+float capabilities) not found, we recommend you install it")
# TODO: Perhaps downloading?
#    set(FFTW_INSTALL_DIR ${CMAKE_BINARY_DIR}/fftw-install)
#    ExternalProject_Add(fftw
#        URL http://www.fftw.org/fftw-3.3.10.tar.gz
#        PREFIX fftw-src
#        CONFIGURE_COMMAND ./configure --prefix=${FFTW_INSTALL_DIR} --enable-float
#        BUILD_COMMAND make -j4
#        INSTALL_COMMAND make install
#    )
#    set(FFTW_INCLUDE_DIRS ${FFTW_INSTALL_DIR}/include)
#    set(FFTW_FLOAT_LIB ${FFTW_INSTALL_DIR}/lib/libfftw3f.so)
#    set(FFTW_FOUND TRUE)
#    add_definitions(-DHAS_FFTW)
endif()

# Installing FFTW https://www.fftw.org/fftw3_doc/Installation-on-Unix.html ---------------------------------------------
# wget https://www.fftw.org/fftw-3.3.10.tar.gz
# tar -xvzf fftw-3.3.10.tar.gz
# cd fftw-3.3.10
# ./configure --enable-float --enable-shared    # --enabled-shared is required for ctypes.util.find_library("fftw3f")
# sudo make
# sudo make install
# ldconfig

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Gtest: ---------------------------------------------------------------------------------------------------------------
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

include_directories(include extern)
add_subdirectory(benchmarks)
