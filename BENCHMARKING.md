# Benchmarking

We provide a master script that setups the entire benchmarking suite for you.

## Setting up Data

To download all the datasets and generate all the indexes needed to run our benchmarking suite, you can use the script [/benchmarks/python_scripts/setup_data.py](/benchmarks/python_scripts/setup_data.py). For this, you need Python 3.11 or higher and install the dependencies in `/benchmarks/python_scripts/requirements.txt`. 

> [!CAUTION]  
> You will need roughly 300GB of disk for ALL the indexes of the datasets used in our paper.

Run the script from the root folder with the script flags `DOWNLOAD` and `GENERATE_IVF` set to `True` and the values in the `ALGORITHMS` array uncommented. You do not need to generate the `ground_truth` for k <= 100 as it is already present. 

You can specify the datasets you wish to create indexes for on the `DATASETS_TO_USE` array in the master script.
```sh
pip install -r ./benchmarks/python_scripts/requirements.txt
python ./benchmarks/python_scripts/setup_data.py
```
The indexes will be created under the `/benchmarks/datasets/` directory.

### Manually downloading data
You can also:
- Manually download all the datasets from a .zip file (~60GB zipped and ~80GB unzipped) [here](https://drive.google.com/file/d/1ei6DV0goMyInp_wFcrbJG3KV40mAPfAa/view?usp=sharing). You must put the unzipped `.hdf5` files inside `/benchmarks/datasets/downloaded`.
- Download datasets individually from [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing). 

Then, run the Master Script with the flag `DOWNLOAD = False`. 

You can specify the datasets you wish to create indexes for on the `DATASETS_TO_USE` array in the master script.


### Configuring the IVF indexes
Configure the IVF indexes in [/benchmarks/python_scripts/setup_core_index.py](/benchmarks/python_scripts/setup_core_index.py). The benchmarks presented in our publication use `n_buckets = 2 * sqrt(n)` for the number of inverted lists (buckets) and `n_training_points = 50 * n_buckets`. This will create solid indexes fairly quickly.

## Running Benchmarks
Once you have downloaded and created the indexes, you can start benchmarking. 

### Requirements
1. Clang++17 or higher.
2. CMake 3.26 or higher.
3. Set CXX variable. E.g., `export CXX="/usr/bin/clang++-18"`

### Building
We built our scripts with the proper `march` flags. Below are the flags we used for each microarchitecture:
```sh
# GRAVITON4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v2"
# GRAVITON3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v1"
# Intel Sapphire Rapids (256 vectors are used if mprefer-vector-width is not specified)
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=sapphirerapids -mtune=sapphirerapids -mprefer-vector-width=512"
# ZEN4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver4 -mtune=znver4"
# ZEN3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver3 -mtune=znver3"

make
```

On the [/benchmarks/CMakeLists.txt](/benchmarks/CMakeLists.txt) file, you can find which `.cpp` files map to which benchmark.


## Complete benchmarking scripts list

### IVF index searches

- PDX IVF ADSampling: `/benchmarks/BenchmarkPDXADSampling`
- PDX IVF ADSampling + SQ8: `/benchmarks/BenchmarkU8PDXADSampling`
- PDX Two-Level IVF ADSampling: `/benchmarks/BenchmarkIVF2ADSampling`
- PDX Two Level IVF ADSampling + SQ8: `/benchmarks/BenchmarkU8IVF2ADSampling`
- PDX IVF BOND: `/benchmarks/BenchmarkPDXIVFBOND`
- FAISS IVF: `/benchmarks/python_scripts/ivf_faiss.py`

All of these programs have two optional parameters:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.
- `<buckets_nprobe>` to specify the `nprobe` parameter on the IVF index, which controls the recall. If not given or `0`, it will use a series of parameters from 2 to 4096 set in the [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

PDX IVF BOND has an additional third parameter:
- `<dimension_ordering_criteria>`: An integer value. On Intel SPR, we use distance-to-means (`1`). For the other microarchitectures, we use dimension-zones (`5`). Refer to Figure 5 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf).

> [!IMPORTANT]   
> Recall that the IVF indexes must be created beforehand by the `setup_data.py` script.

###  Exact Search
- PDX BOND: ```/benchmarks/BenchmarkPDXBOND```
- USearch: ```python /benchmarks/python_scripts/exact_usearch.py```
- SKLearn: ```python /benchmarks/python_scripts/exact_sklearn.py```
- FAISS: ```python /benchmarks/python_scripts/exact_faiss.py```

All of these programs have one optional parameter:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

PDX BOND has an additional second parameter:
- `<dimension_ordering_criteria>`: An integer value. On exact-search, we always use distance-to-means (`1`). Refer to Figure 5 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf).

**Notes**: Usearch, SKLearn, and FAISS scripts expect the original `.hdf5` files under the `/downloaded` directory.  Furthermore, they require their respective Python packages (`pip install -r ./benchmarks/python_scripts/requirements.txt`).

## Kernels Experiment
Visit our playground for PDX vs SIMD kernels [here](./benchmarks/bench_kernels)

## SIGMOD'25
Check the `sigmod` branch.