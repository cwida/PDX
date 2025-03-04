# PDX: A Vertical Data Layout for Vector Similarity Search

[PDX](https://ir.cwi.nl/pub/35044/35044.pdf) is a **vertical** data layout for vectors that store the dimensions of different vectors together. In a nutshell, PDX is PAX for vector similarity search.

<p align="center">
        <img src="./benchmarks/results/pdx-layout.png" alt="PDX Layout" style="{max-height: 150px}">
</p>


### What are the benefits of storing vectors vertically?

- ✂️ Efficient **pruning** of dimensions with partial distance calculations. 
- ⚡ Up to [**7x faster**](#ivf-indexes) **IVF queries** than FAISS+AVX512.
- ⚡ Up to [**13x faster**](#exact-search--ivf) **exhaustive search** thanks to pruning.
- ⚡ **Distance kernels** in PDX are up to [**1.6x faster**](#no-pruning-and-no-index) than the `float32` kernels in USearch and FAISS. 
  - *Why?* Distance kernels in PDX are free of dependencies and have fewer LOAD/STORE operations. 
- Distance kernels can **auto-vectorize efficiently** without explicit SIMD for `float32`.
- Distance kernels on **small vectors** (`d < 16`) are up to **8x faster** than SIMD kernels in [SimSIMD](https://github.com/ashvardanian/SimSIMD).
- More efficient compressed representations of vectors (WIP).


## Contents
- [Pruning in a Nutshell](#pruning-in-a-nutshell)
- [Quickstart](#quickstart)
- [Use cases (comparison with FAISS)](#use-cases)
- [Roadmap](#roadmap)

## Pruning in a nutshell

*Pruning* means avoiding checking *all* the dimensions of a vector to determine if it will make it onto the KNN of a query. Pruning speedup vector similarity search as (i) **less data** must be fetched and (ii) **fewer computations** must be done.

However, pruning methods that do partial distance calculations have a hard time being on par with SIMD-optimized kernels like the ones in [FAISS](https://github.com/facebookresearch/faiss/) and [SimSIMD](https://github.com/ashvardanian/SimSIMD). 

**Only thanks to the PDX layout**, do pruning methods outperform SIMD-optimized kernels in all CPU microarchitectures (Zens, Intels, Gravitons). This is because, in PDX, distance calculations are efficient even in small vector segments and vectors with a low dimensionality. Also, the evaluation of the pruning predicate is not interleaved with distance calculations.

Click [here](#use-cases) to see some **benchmarks** of PDX vs FAISS. The complete benchmarks are available in [our publication](https://ir.cwi.nl/pub/35044/35044.pdf).

Pruning algorithms are **especially effective** when:
- Vectors are of high dimensionality (`d > 512`)
- High recalls are needed (`> 0.90`)
- Exact results are needed
- `k` is relatively low (`k=10,20`)

The latest **pruning algorithms** based on partial distance calculations are [ADSampling](https://github.com/gaoj0017/ADSampling/), [DDC](https://github.com/mingyu-hkustgz/Res-Infer) (previously named BSA), and [DADE](https://github.com/Ur-Eine/DADE). All of these rely on rotating the vector collection to prune effectively. PDX+ADSampling works great in most scenarios.

Alongside PDX, we also introduce **PDX-BOND**, a simpler and exact pruning algorithm that does not need to rotate the vectors (based on [BOND](https://dl.acm.org/doi/pdf/10.1145/564691.564729)). 


## Quickstart
You can quickly try PDX with your data using our Python bindings and [examples](/examples). We have implemented PDX on Flat **IVF indexes** and **exhaustive search** settings.
### Prerequisites
- Python 3.11 or higher
- [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) with Python Bindings
- Clang++17 or higher
- CMake 3.26 or higher

### Steps
1. Clone the repository and init submodules (`Eigen` for efficient matrix operations and `pybind11`)
```sh
git clone https://github.com/cwida/PDX
git submodule init
git submodule update
```
2. Install Python dependencies and `pdxearch` Python bindings. 
```sh
export CXX="/usr/bin/clang++-18"  # Set proper CXX first
pip install -r requirements.txt
python setup.py clean --all
python -m pip install .
```
3. Run the examples under `/examples`
```sh
# Creates an IVF index with FAISS on random data
# Then, it compares the search performance of PDXearch and FAISS
python ./examples/pdxearch_simple.py
```
For more details on the available examples and how to use your own data, refer to [/examples/README.md](./examples/README.md). 

> [!NOTE]
> We heavily rely on [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) to create the underlying IVF indexes. 

## Use Cases
We present some single-threaded **benchmarks** from our [examples](/examples) against FAISS for some use cases in which PDX shines. We use `r7iz.2xlarge` (Intel SPR) and `r8g.2xlarge` (Graviton 4) AWS instances.  
### IVF indexes
PDX paired with the pruning algorithm [ADSampling](https://github.com/gaoj0017/ADSampling/) on IVF indexes works great in most scenarios with less than 0.001 recall loss. The higher the dimensionality, the higher the gains from pruning. The following benchmarks are from [/examples/pdxearch_ivf.py](./examples/pdxearch_ivf.py). 

| Avg. query time <br> [<ins>Intel SPR</ins> \| r7iz.2x] | FAISS AVX512 <br> R@10: 0.99 · 0.95 · 0.90 | PDXearch           | Improvement            |
|--------------------------------------------------------|-----------------------------------------|--------------------|------------------------|
| DBPedia · d=1536 · 1M                                | 53.6 · 18.0 · 7.7 ms                    | 7.4 · 3.7 · 2.4 ms | **7.2x · 4.9x · 3.2x** |
| arXiv · d=768 · 2.25M                                | 25.9 · 7.7 · 3.5                        | 6.2 · 2.8 · 1.7    | **4.2x · 2.8x · 2.1x** |
| SIFT · d=128 · 1M                                    | 1.7 · 0.6 · 0.4                         | 1.0 · 0.4 · 0.3    | **1.7x · 1.5x · 1.3x** |

| Avg. query time <br> [<ins>Graviton 4</ins> \| r8g.2x] | FAISS SVE <br> R@10: 0.99 · 0.95 · 0.90 | PDXearch           | Improvement            |
|-------------------------------------------------------------|-----------------------------------|--------------------|------------------------|
| DBPedia · d=1536 · 1M                                     | 47.1 · 18.4 · 6.7 ms              | 7.1 · 4.1 · 2.5 ms | **6.6x · 4.5x · 2.7x** |
| arXiv · d=768 · 2.25M                                     | 25.3 · 7.0 · 3.2                  | 5.9 · 2.7 · 1.7    | **4.3x · 2.6x · 1.9x** |
| SIFT · d=128 · 1M                                         | 1.1 · 0.5 · 0.3                   | 0.7 · 0.4 · 0.2    | **1.6x · 1.3x · 1.3x** |

> [!NOTE]   
> On these benchmarks: (i) Both FAISS and PDXearch are scanning exactly the same vectors. (ii) The recall loss of ADSampling is always less than 0.001.


### Exact search + IVF
In PDX, building an IVF index can significantly improve exact search speed (thanks to the reliable pruning), even if the clustering is not super optimized. The following benchmarks are from [/examples/pdxearch_ivf_exhaustive.py](./examples/pdxearch_ivf_exhaustive.py).

| Avg. query time<br>[<ins>Intel SPR</ins> \| r7iz.2x] | FAISS <br> AVX512 | PDXearch | Improvement |
|-------------------------------------------------------------------|---------------|----------|-------------|
| DBPedia · d=1536 · 1M                                           | 411.7 ms      | 34.9 ms  | **11.8x**   |
| GIST · d=960 · 1M                                               | 252.9         | 27.6     | **9.1x**    |
| arXiv · d=768 · 2.25M                                           | 454.5         | 41.4     | **11.0x**   |
| SIFT · d=128 · 1M                                               | 34.4          | 6.5      | **5.3x**    |

| Avg. query time<br>[<ins>Graviton 4</ins> \| r8g.2x] | FAISS <br> SVE | PDXearch | Improvement |
|-----------------------------------------------------------|------------|----------|-------------|
| DBPedia · d=1536 · 1M                                   | 277.2 ms   | 21.7 ms  | **12.8x**   |
| GIST · d=960 · 1M                                       | 170.9      | 19.4     | **8.8x**    |
| arXiv · d=768 · 2.25M                                   | 306.0      | 27.2     | **11.3x**   |
| SIFT · d=128 · 1M                                       | 21.3       | 4.7      | **4.5x**    |

### Exact search without an index
Use **PDX+BOND**, our pruning algorithm. Here, vectors are not transformed, and we do not use any additional index. Gains vary depending on the dimensions distribution. The following benchmarks are from [/examples/pdxearch_exact_bond.py](./examples/pdxearch_exact_bond.py).

| Avg. query time<br>[<ins>Intel SPR</ins> \| r7iz.2x] | FAISS <br> AVX512 | PDXearch | Improvement |
|------------------------------------------------------|---------------|----------|-------------|
| DBPedia · d=1536 · 1M                                | 374 ms        | 216 ms   | **1.7x**    |
| arXiv · d=768 · 2.25M                                | 422           | 212      | **2.0x**    |
| MSong · d = 420 · 1M                                 | 117           | 15       | **7.8x**    |
| SIFT · d=128 · 1M                                    | 31            | 10       | **3.1x**    |

| Avg. query time <br> [<ins>Graviton 4</ins> \| r8g.2x] | FAISS <br> SVE | PDXearch | Improvement |
|--------------------------------------------------------|------------|----------|-------------|
| DBPedia · d=1536 · 1M                                  | 278 ms     | 139 ms   | **2.0x**    |
| arXiv · d=768 · 2.25M                                  | 305        | 155      | **2.0x**    |
| MSong · d = 420 · 1M                                   | 70         | 15       | **4.7x**    |
| SIFT · d=128 · 1M                                      | 22         | 11       | **2.0x**    |


### No pruning and no index
PDX distance kernels are also faster than the state-of-the-art SIMD kernels in all major architectures, only relying on auto-vectorization (for `float32`). For a detailed explanation of how these distance kernels work, check Figure 3 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf). The following benchmarks are from [/examples/pdx_brute.py](./examples/pdx_brute.py).

| Avg. query time<br>[<ins>Intel SPR</ins> \| r7iz.2x] | USearch | FAISS <br> AVX512 | PDXearch | Improvement     |
|------------------------------------------------------|---------|---------------|----------|-----------------|
| GIST · d=960 · n=1M                                  | 260 ms  | 247 ms        | 208 ms   | **1.3x · 1.2x** |
| STL · d=9216 · n=90K                                 | 223     | 220           | 176      | **1.3x · 1.3x** |

| Avg. query time<br>[<ins>Graviton 4</ins> \| r8g.2x] | USearch | FAISS <br> SVE | PDXearch | Improvement     |
|------------------|---------|------------|----------|-----------------|
| GIST · d=960 · n=1M   | 203 ms  | 172 ms     | 124 ms   | **1.6x · 1.4x** |
| STL · d=9216 · n=90K   | 175     | 160        | 109      | **1.6x · 1.5x** |


> [!NOTE]   
> Refer to [our publication](https://ir.cwi.nl/pub/35044/35044.pdf) for the complete benchmarks of PDXearch done in 4 microarchitectures: Intel SPR, Zen 4, Zen 3, and Graviton 4. Here, you will also find a comparison against ADSampling in the horizontal layout (`.fvecs` layout). Furthermore, you will find details on how we adapted pruning algorithms to work in PDX.

## Roadmap
- **Compression**: The vertical layout opens opportunities for compression as indexing algorithms group together vectors that share some numerical similarity within their dimensions. The next step on PDX is to apply our scalar quantization algorithm [LEP](https://homepages.cwi.nl/~boncz/msc/2024-ElenaKrippner.pdf), which uses database compression techniques ([ALP](https://github.com/cwida/alp)) to compress vectors with higher compression ratios and less information loss.
- **More data types**: For compressed vectors, we need to implement vertical distance kernels on vectors of variable bit size.
- **PDX in HNSW**: For this, we need a layout similar to the ones proposed in [Starling](https://dl.acm.org/doi/pdf/10.1145/3639269) or [AiSAQ](https://arxiv.org/pdf/2404.06004), in which neighborhoods of the graph are stored and fetched in blocks.
- Improve code readability and usability.
- Add a testing framework.
- Add DDC or DADE algorithms to the Python Bindings.
- Implement multi-threading capabilities.

> [!IMPORTANT]   
> PDX is an ongoing research project. In its current state, it is not production-ready code.


## Benchmarking
To run our benchmark suite in C++, refer to [BENCHMARKING.md](./BENCHMARKING.md).

## SIGMOD
The code used for the experiments presented at SIGMOD'25 can be found in the `sigmod` branch.
