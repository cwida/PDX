# PDX Kernels Playground
With the `kernels.py` script, you can test the raw performance of different PDX-based kernels against SIMD ones on different scenarios and with different datasets (downloaded or randomly generated). 

The library will automatically detect and use NEON and AVX512 accordingly. It does not yet support AVX2 or SVE. 

The used KNN algorithm is brute-force with a partial-sort at the end. All benchmarks are single-threaded, and measurements are end-to-end.

## Install and Compile

#### Install UV
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Create VENV and Install libraries
```sh
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Set clang++ alias (if not set already)
```sh
alias clang++='/usr/bin/clang++-18'
source ~/.bashrc
```

#### Compile Kernels
```sh
clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -o ./kernels.dylib ./kernels.cpp
```

## Run
```sh
# 16384 vectors, 100 queries, d=1024, f32
uv run --script kernels.py --count 16384 --ndims 1024 --k 1 --query_count 100 --warmup 10

# 16384 vectors, 100 queries, d=1024, u8
uv run --script kernels.py --count 16384 --ndims 1024 --k 1 --query_count 100 --warmup 10 --dtype u8

# simplewiki-openai-3072-normalized dataset, 100 queries, u8
uv run --script kernels.py --k 1 --query_count 100 --warmup 1 --dataset simplewiki-openai-3072-normalized --dtype u8 
```

Parameters:
- `count`: Number of vectors.
- `query_count`: Number of queries.
- `ndims`: Number of dimensions.
- `k`: Number of nearest neighbours. (default=`1`, as we are more interested in the distance calculation performance)
- `output`: A path to save the benchmarking results in a .csv format.
- `dtype`: `f32` or `u8` (default=`f32`)
- `warmup`: Number of search repetitions to warm up the cache.
- `dataset`: Name of the dataset to use (searching for a .hdf5 file within ./benchmarks/datasets/downloaded/). 
  - If used, overrides `count` and `ndims` to the dataset configuration. 
  - `query_count` is overridden only if the dataset query count is less than the specified `query_count`. 
  - If used alongside `dtype=u8`, does a min-max scalar quantization. 
  - If used, ground truth is not checked. 
  - `count` is floored to the nearest multiple of 64 (for PDX to work without tails).

You may need to do the following to avoid an error on cppyy in MacOS:
```sh
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/zstd/lib:$DYLD_LIBRARY_PATH
```

 