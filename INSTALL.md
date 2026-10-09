# Installation

### PDX needs:
- Clang 17 (MSVC 2022 on Windows), CMake 3.26
- Python 3 (only for Python bindings)

Once you have these requirements, you can install the Python Bindings

<details>
<summary> <b> Installing Python Bindings </b></summary>

```sh
git clone https://github.com/cwida/PDX
cd PDX

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
- `-DSKMEANS_EXECUTOR`: the thread pool behind the parallel loops: `forkunion` (default), `openmp` or `serial`. You can also pass your own pool through `PDXIndexConfig::executor`.
- `-DSKMEANS_GEMM`: the matrix multiplication backend: `auto` (default: Apple Accelerate on macOS, Eigen elsewhere), `eigen`, `accelerate` or `blas` (see [Using an external BLAS](#using-an-external-blas-optional)).
- `-DPDX_COMPILE_TESTS` / `-DPDX_COMPILE_BENCHMARKS`: compile the C++ tests (`make tests`, run with `ctest`) and benchmarks (`make benchmarks`). Both default to `OFF`.


## Step by Step
* [Installing Clang](#installing-clang)
* [Installing CMake](#installing-cmake)
* [Using an external BLAS (optional)](#using-an-external-blas-optional)
* [Using OpenMP (optional)](#using-openmp-optional)
* [Installing FFTW (optional)](#installing-fftw-optional)
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

## Using an external BLAS (optional)
By default, PDX multiplies matrices with Eigen (fetched by CMake), or with Apple Accelerate on macOS.

### MacOS
**Silicon Chips (M1 to M5)**: You don't need to do anything special. We automatically use [Apple Accelerate](https://developer.apple.com/documentation/accelerate), which uses the [AMX](https://github.com/corsix/amx) unit. 

**Intel Chips (older Macs)**: We use Apple Accelerate as well.

### Linux
In our benchmarks the default Eigen performs roughly on par with OpenBLAS (on par), MKL (5-10% slower e2e), and BLIS (on par). To use these instead, configure with `-DSKMEANS_GEMM=blas`. **BLAS must be single-threaded and thread-safe**:
- **Intel MKL**: detected automatically and linked in its sequential version.
- **OpenBLAS**: if no BLAS is found, CMake builds a suitable OpenBLAS from source. To build one yourself:
  ```sh
  git clone https://github.com/OpenMathLib/OpenBLAS.git
  cd OpenBLAS
  make -j$(nproc) DYNAMIC_ARCH=1 USE_THREAD=0 USE_LOCKING=1
  make PREFIX=/usr/local install
  ldconfig
  ```
  With a multi-threaded OpenBLAS (e.g. from `apt`), set `OPENBLAS_NUM_THREADS=1`.
- **AMD AOCL BLIS**: link the single-threaded library (`libblis.so`, not `libblis-mt.so`).

To force a specific library, pass its path:
```sh
# C++
cmake . -DSKMEANS_GEMM=blas -DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis.so

# Python Installation
pip install --force-reinstall . -C cmake.args="-DSKMEANS_GEMM=blas;-DBLAS_LIBRARIES=/opt/amd-blis/lib/libblis.so"
```

## Using OpenMP (optional)
The default thread pool is [ForkUnion](https://github.com/ashvardanian/ForkUnion), fetched by CMake. To use OpenMP instead, install it and configure with `-DSKMEANS_EXECUTOR=openmp`.

### Linux
Most distributions come with OpenMP, or you can install it with:
```sh
sudo apt-get install libomp-dev
```

### MacOS
```sh 
brew install libomp
```

## Installing FFTW (optional)
[FFTW](https://www.fftw.org/fftw3_doc/Installation-on-Unix.html) will give you better performance in very high-dimensional datasets (d > 1024). 

```sh
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xvzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --enable-float --enable-shared
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

