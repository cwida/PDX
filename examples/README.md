# Examples
`pdx_simple.py` shows a plug-ang-play example that creates a random collection with scikit-learn. The rest of the examples read vectors from the `.hdf5` format.

## Downloading the data
Our examples look for `.hdf5` files in `/benchmarks/datasets/downloaded`. These datasets follow the convention used in the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks/) project. One `.hdf5` file with two datasets: `train` and `test`. We have a few ways in which you can download the data we used:
- Download and unzip ALL the `.hdf5` datasets from [here](https://drive.google.com/file/d/1ei6DV0goMyInp_wFcrbJG3KV40mAPfAa/view?usp=sharing) (~60GB zipped and ~80GB unzipped).
- Download datasets individually from [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing). 
- Run the script [`/benchmarks/python_scripts/setup_data.py`](/benchmarks/python_scripts/setup_data.py) from the root folder with the script flag `DOWNLOAD = True`. This will download and unzip ALL the `.hdf5` datasets. Make sure you set all the other flags to `False` and comment the elements inside the `ALGORITHMS` array.


## Examples description

- `pdx_simple.py`: Pruned search with [ADSampling](https://github.com/gaoj0017/ADSampling/) with an IVF index (built with FAISS). Plug & Play example that uses a random collection of vectors.
- `pdx_2l_ivf.py`: Pruned search + ADSampling with a **Two-Level IVF index** (built with FAISS). The recall is controlled with the `nprobe` parameter.
- `pdx_2l_ivf_8bit.py`: Pruned search + ADSampling with a **Two-Level IVF index** (built with FAISS) using 8-bit Scalar Quantization. The recall is controlled with the `nprobe` parameter.
- `pdx_ivf.py`: Pruned search + ADSampling with a **vanilla IVF index** (built with FAISS). The recall is controlled with the `nprobe` parameter.
- `pdx_ivf_exhaustive.py`: Pruned search with ADSampling with a IVF index (built with FAISS). This example explore all the clusters (therefore, it is an exhaustive search). This lets the pruning strategy shine and get **up to 13x speedup**.
- `pdx_noindex.py`: Pruned search with ADSampling on the entire collection (no index). This produces nearly exact results. 
- `pdx_noindex_bond.py`: Pruned search with BOND on the entire collection (no index). This produces exact results and does not need a preprocessing of the data.
- `pdx_persist.py`: Example to store the PDX index and the metadata of ADSampling in a file to use later.
