# Examples

`pdx_simple.py` shows a plug-and-play example that creates a random collection with scikit-learn. The rest of the examples read vectors from the `.hdf5` format that you can download [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing).

## Running the examples

Check [INSTALL.md](/INSTALL.md) for installation instructions. Then, run the examples:

```sh
python ./examples/pdx_simple.py
```

## Downloading the data
Our examples look for `.hdf5` files in `/benchmarks/datasets/downloaded`. These datasets follow the convention used in the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks/) project. One `.hdf5` file with two datasets: `train` and `test`. We have a few ways in which you can download the data we used:
- Download and unzip ALL the `.hdf5` datasets from [here](https://drive.google.com/file/d/1ei6DV0goMyInp_wFcrbJG3KV40mAPfAa/view?usp=sharing) (~60GB zipped and ~80GB unzipped).
- Download datasets individually from [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing).
- Run the script [`/benchmarks/python_scripts/setup_data.py`](/benchmarks/python_scripts/setup_data.py) from the root folder with the script flag `DOWNLOAD = True`. This will download and unzip ALL the `.hdf5` datasets. Make sure you set all the other flags to `False` and comment the elements inside the `ALGORITHMS` array.


## Examples description

- **`pdx_simple.py`**: Plug-and-play example using random vectors generated with scikit-learn. Builds a single-level IVF index (`IndexPDXIVF`) and runs approximate nearest-neighbor queries. No external datasets required.

- **`pdx_ivf.py`**: Single-level IVF index (`IndexPDXIVF`, F32 precision) on vector embeddings. Demonstrates building an index, querying with a configurable `nprobe`, and inspecting index properties like cluster count and in-memory size.

- **`pdx_ivf_exhaustive.py`**: Exhaustive search using `nprobe=0` (visits all clusters). Compares PDX pruned search against brute-force FAISS (`IndexFlatL2`) to show the speedup from ADSampling pruning even without approximate search.

- **`pdx_tree_sq8.py`**: Two-level hierarchical IVF index with 8-bit scalar quantization (`IndexPDXIVFTreeSQ8`). The two-level structure allows pruning to happen when finding the most promising clusters. Recall is controlled with `nprobe`.

- **`pdx_filtered.py`**: Filtered (predicated) search using `IndexPDXIVFTreeSQ8`. Demonstrates how to pass a set of allowed row IDs to `filtered_search()`, restricting results to only those vectors. Includes a correctness check verifying that returned IDs are a subset of the allowed set.

- **`pdx_persist.py`**: Save and load a PDX index to/from disk. Builds an index, saves it with `index.save()`, then reloads it with `load_index()` and queries the restored index.
