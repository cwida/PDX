# PDX: A Vertical Data-Layout for Vector Similarity Search

[PDX](https://ir.cwi.nl/pub/35044/35044.pdf) is a **vertical** data layout for vectors that stores together the dimensions of different vectors. In a nutshell, vectors are partitioned by dimensions.

### What are the benefits of storing vectors vertically?

- Distance calculations happen dimension-by-dimension. This (i) auto-vectorize efficiently without explicit SIMD for `float32`, (ii) is 40% faster in avg. than regular SIMD kernels and (iii) makes distance calculations on small vectors (`d < 16`) up to 8x faster.
- In PDX, a search can efficiently **prune** dimensions with partial distance calculations. For instance, pairing PDX with the pruning algorithm [ADSampling](https://github.com/gaoj0017/ADSampling/), achieves 2x-7x faster IVF queries than FAISS+AVX512.
- More efficient compressed representation of vectors (WIP)

## Pruning in a nutshell

*Pruning* means avoiding checking *all* the dimensions of a vector to determine if it will make it onto the KNN of a query. 

*Pruning* speedup vector search as (i) less data must be fetched and (ii) fewer computations must be done.

However, pruning methods that do partial distance calculations have a hard time to be on-par to SIMD optimized kernels like the ones in [FAISS](https://github.com/facebookresearch/faiss/) and [SimSIMD](https://github.com/ashvardanian/SimSIMD). 

**Thanks to the PDX layout**, pruning methods outperform SIMD optimized kernels. This is because in PDX, distance kernels efficiency are not limited by the number of dimensions inspected (which is low in pruning methods). And, the evaluation of the pruning threshold is not interleaved with distance calculations.

Pruning is **especially effective** when:
- Vectors are of high dimensinality (`d > 512`) 
- `k` is low (`k=10,20,30`) 
- Targetting high recalls (`> 0.85`)
- Exact results are needed

We refer to the recent research done on pruning algorithms with partial distance calculations: [ADSampling](https://github.com/gaoj0017/ADSampling/), [BSA](https://github.com/mingyu-hkustgz/Res-Infer), [DADE](https://github.com/Ur-Eine/DADE).

## Try it out
### Prerequisites
- Python 3.11 or higher
- FAISS with Python Bindings
- Clang++ 17 or higher
- CMake 3.20 or higher

### Steps
1. Clone the repository and init submodules (`Eigen` for efficient matrix operations and `pybind11`)
```sh
git clone https://github.com/cwida/PDX
git submodule init
git submodule update
```
2. Install Python dependencies and `pdxearch` Python bindings. 
```sh
export CXX="/usr/bin/clang++-18"
pip install -r requirements.txt
python setup.py clean --all
python -m pip install .
```
3. Run the examples under `./examples`
```sh
# Creates an IVF index with FAISS on random data
# Then, it perform queries on it with PDXearch and FAISS
python ./examples/pdxearch_simple.py
```
For more details on each example we provide, and how to use your own data, refer to [/examples/README.md](./examples/README.md). 

### Notes
- We heavily rely on FAISS to create the underlying IVF indexes. 
- PDX is an ongoing research project. In its current state, it is not production-quality code.

## Which pruning algorithm should I use?

**PDX+ADSampling** is the best option if you can tolerate little error. Expect speedups of up to 2x for low dimensional vectors (`d < 200`) and up to 10x for high dimensional vectors. See EXAMPLEXX.py

*some table with data...*

**PDX+BOND** is the best option if you need exact answers or/and you cannot transform the original vectors. Expect speedups of 1.5-4x for exact search, without any additional index. See EXAMPLEXX.py

*some table with data...*

If you want exact search, you can find huge benefits by building an IVF index and visit all the buckets. See EXAMPLEXX.py

*some table with data...*

## Roadmap
- **Compression**: The vertical layout opens opportunities to compress vectors better as indexing algorithms group together vectors that share some numerical similarity within their dimensions. A next step on PDX is to apply our scalar quantization algorithm [LEP](https://homepages.cwi.nl/~boncz/msc/2024-ElenaKrippner.pdf) that uses database compression techniques ([ALP](https://github.com/cwida/alp)) to compress vectors at higher compression ratios with little information loss.
- **More data types**: For compressed vectors, we need to implement vertical distance kernels on variable-bit size vectors.
- Improve code readibility and usability.
- Add a testing framework
- Add BSA algorithm to the Python Bindings

## Benchmarking
To run our benchmarks in C++, refer to [BENCHMARKING.md](./BENCHMARKING.md).

## SIGMOD 2025
The code used for the experiments presented at SIGMOD can be found in the `sigmod-2025` branch.


