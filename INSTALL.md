# Installation

### PDX needs:
- Clang 17, CMake 3.26
- OpenMP
- A BLAS implementation
- Python 3 (only for Python bindings)

Once you have these requirements, you can install the Python Bindings

<details>
<summary> <b> Installing Python Bindings </b></summary>

```sh
git clone https://github.com/cwida/PDX
cd PDX
git submodule update --init

# Create a venv if needed
python -m venv ./venv
source venv/bin/activate

# Set proper clang compiler if needed
export CXX="/usr/bin/clang++-18" 

pip install .
```
</details>

<details>
<summary> <b> Compiling C++ tests and benchmarks from source </b></summary>

```sh
git clone https://github.com/cwida/PDX
cd PDX
git submodule update --init

# Set proper clang compiler if needed
export CXX="/usr/bin/clang++-18"

cmake . -DPDX_COMPILE_TESTS=ON -DPDX_COMPILE_BENCHMARKS=ON
make -j$(nproc) tests benchmarks
```
</details>

### Compilation knobs
All of these are CMake cache variables (`cmake . -D<KNOB>=<value>`). For `pip install .`, pass them with `-C cmake.args="-D<KNOB>=<value>"`.
- `-DPDX_MARCH`: `-march` value to use during PDX compilation (default=`native`). An empty string disables `-march`.
- `-DPDX_PORTABLE`: `ON` replaces `-march` with portable SIMD flags (`-mavx2 -mfma` on x86_64, plain `-O3` elsewhere). Meant for wheel builds; also settable through the `PDX_PORTABLE` environment variable during `pip install .`.
- `-DPDX_SKIP_FFTW`: `ON` skips the optional FFTW dependency entirely (also settable through the `PDX_SKIP_FFTW` environment variable during `pip install .`).
- `-DBLAS_LIBRARIES`: Full path name of the BLAS library to use. Useful if you want to link PDX against a different BLAS implementation. For example, if you have installed AMD BLIS: `-DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis-mt.so`
- `-DPDX_COMPILE_TESTS` / `-DPDX_COMPILE_BENCHMARKS`: compile the C++ tests (`make tests`, run with `ctest`) and benchmarks (`make benchmarks`). Both default to `OFF`.

> [!TIP]
> BLAS is critical to achieve high performance. On Linux, we recommend [installing OpenBLAS from source](#installing-blas).


## Step by Step
* [Installing Clang](#installing-clang)
* [Installing CMake](#installing-cmake)
* [Installing OpenMP](#installing-openmp)
* [Installing BLAS](#installing-blas)
* [Installing FFTW](#installing-blas) [optional]
* [Troubleshooting](#troubleshooting)

## Installing Clang
We recommend LLVM
### Linux
```sh 
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 18
```

### MacOS
```sh 
brew install llvm
```

## Installing CMake
### Linux
```sh 
sudo apt update
sudo apt install make
sudo apt install cmake
```

### MacOS
```sh 
brew install cmake
```

## Installing OpenMP

### Linux
Most distributions come with OpenMP, or you can install it with:
```sh
sudo apt-get install libomp-dev
```

### MacOS
```sh 
brew install libomp
```


## Installing BLAS

BLAS is extremely important to achieve high performance. We recommend [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). 

### Linux
Most distributions come with [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), or you may have already installed OpenBLAS via `apt`. **THIS IS SLOW**. We recommend installing OpenBLAS from source with the commands below.

```sh
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) DYNAMIC_ARCH=1 USE_OPENMP=1 NUM_THREADS=128
make -j$(nproc) PREFIX=/usr/local install
ldconfig
```

### MacOS
**Silicon Chips (M1 to M5)**: You don't need to do anything special. We automatically detect [Apple Accelerate](https://developer.apple.com/documentation/accelerate) that uses the [AMX](https://github.com/corsix/amx) unit. 

**Intel Chips (older Macs)**: Install OpenBLAS as detailed above.

## Installing FFTW
[FFTW](https://www.fftw.org/fftw3_doc/Installation-on-Unix.html) will give you better performance in very high-dimensional datasets (d > 1024). 

```sh
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xvzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --enable-float --enable-shared --enable-openmp
sudo make -j$(nproc)
sudo make install
ldconfig
```

## Troubleshooting

### Python bindings installation fails

Error:
```
Could NOT find Python (missing: Development.Module) 
    Reason given by package:
        Development: Cannot find the directory "/usr/include/python3.12"
```

Solution: Install `python-dev` package:

```sh
sudo apt install python3-dev
```

### I get a bunch of `warnings` when compiling PDX

If you see a lot of warnings like this one:
```warning: ignoring ‘#pragma clang loop’```

You are using GCC instead of Clang. If you installed Clang, you can set the correct compiler by doing the following:
```sh
export CXX="/usr/bin/clang++-18" # Linux

export CXX="/opt/homebrew/opt/llvm/bin/clang++" # MacOS
```

### Does PDX use SIMD?
Yes. We have optimizations for AVX512, AVX2, and NEON. You don't need to do anything special to activate these. If your machine doesn't have any of these, we rely on scalar code. 

