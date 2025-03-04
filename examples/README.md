# Examples
A Plug & Play example is given in `pdxearch_simple.py`. This example creates a random collection of vectors with scikit-learn. The rest of the examples read vector data from a `.hdf5` file.

## Downloading the data
Our Python bindings expect Numpy matrices as input. However, the provided examples read vectors in a `.hdf5` format expected to be located at `/benchmarks/datasets/downloaded`. These datasets follow the convention used in the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks/) project. One `.hdf5` file with two datasets: `train` and `test`. We have a few ways in which you can download the data we used:
- Download and unzip ALL the `.hdf5` datasets manually from [here](https://drive.google.com/file/d/1I8pbwGDCSe3KqfIegAllwoP5q6F4ohj2/view?usp=sharing) (~25GB zipped and ~40GB unzipped).
- Download datasets individually from [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing). 
- Run the script [`/benchmarks/python_scripts/setup_data.py`](/benchmarks/python_scripts/setup_data.py) from the root folder with the script flag `DOWNLOAD = True`. This will download and unzip ALL the `.hdf5` datasets (~25GB zipped and ~40GB unzipped). Make sure you set all the other flags to `False` and comment the elements inside the `ALGORITHMS` array.

You can, of course, change each example to read your own data.



## Examples description

This collection of examples uses the algorithms exposed in our Python bindings. 

- `pdx_brute.py`: PDX kernels (without pruning). The full scan on vertical kernels shines when D is high, as the tight loops of the kernel avoid additional LOAD/STORE operations and are free of dependencies. Refer to [Figure 3 in our publication](https://ir.cwi.nl/pub/35044/35044.pdf).
- `pdxearch_simple.py`: PDXearch (pruned search) + [ADSampling](https://github.com/gaoj0017/ADSampling/) with an IVF index (built with FAISS). Plug & Play example that uses a random collection of vectors.
- `pdxearch_exact.py`: PDXearch (pruned search) + ADSampling on the entire collection (no index). This produces virtually exact results. In our experiments, the recall loss due to ADSampling hypothesis testing was never higher than 0.001.
- `pdxearch_exact_bond.py`: PDXearch (pruned search) + BOND on the entire collection (no index). This produces exact results. 
- `pdxearch_ivf.py`: PDXearch (pruned search) + ADSampling with an IVF index (built with FAISS). The recall is controlled with the `nprobe` parameter.
- `pdxearch_ivf_exhaustive.py`: Exact search using PDXearch (pruned search) + ADSampling with an IVF index (built with FAISS). We can do an exact search by exploring all the buckets. This lets the pruning strategy shine and get **up to 13x speedup**. This produces virtually exact results. In our experiments, the recall loss due to ADSampling hypothesis testing was never higher than 0.001.
- `pdxearch_ivf_exhaustive_bond.py`: Exact search using PDXearch (pruned search) + BOND with an IVF index (built with FAISS). We can do an exact search by exploring all the buckets. This produces exact results.
- `pdxearch_persist.py`: Example to store the PDX index and the metadata of ADSampling in a file to use later

Note that as part of our research, we also ran benchmarks of PDXearch against the pruning algorithms [ADSampling](https://github.com/gaoj0017/ADSampling/) and [BSA](https://github.com/mingyu-hkustgz/Res-Infer) (renamed to [DDC](https://arxiv.org/pdf/2404.16322)) on the N-ary/horizontal layout. These are not available to use directly in our Python bindings. Refer to [BENCHMARKING.md](/BENCHMARKING.md).