# Benchmarks

We present single-threaded **benchmarks** against FAISS+AVX512 on an `r7iz.xlarge` (Intel Sapphire Rapids) instance. 

### Two-Level IVF (IVF<sub>2</sub>) ![](https://img.shields.io/badge/Fastest%20search%20on%20PDX-red)
IVF<sub>2</sub> tackles a bottleneck of IVF indexes: finding the nearest centroids. By clustering the original IVF centroids, we can use PDX to quickly scan them (thanks to pruning) without sacrificing recall. This achieves significant throughput improvements when paired with `8-bit` quantization.

<p align="center">
        <img src="./benchmarks/results/ivf2-intel.png" alt="PDX Layout" style="{max-height: 150px}">
</p>

### Vanilla IVF
Here, PDX, paired with the pruning algorithm ADSampling on `float32`, achieves significant speedups.

<p align="center">
        <img src="./benchmarks/results/ivf-intel.png" alt="PDX Layout" style="{max-height: 150px}">
</p>


### Exhaustive search + IVF
An exhaustive search scans all the vectors in the collection. Having an IVF index with PDX can **EXTREMELY** accelerate this without sacrificing recall, thanks to the reliable pruning of ADSampling.

<p align="center">
        <img src="./benchmarks/results/ivf-exhaustive-intel.png" alt="PDX Layout" style="{max-height: 150px}">
</p>

The key observation here is that thanks to the underlying IVF index, the exhaustive search starts with the most promising clusters. A tight threshold is found early on, which enables the quick pruning of most candidates.

### No pruning and no index
Even without pruning, PDX distance kernels can be faster than SIMD ones in most CPU microarchitectures. For detailed information, check Figure 3 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf). You can also try it yourself in our playground [here](./benchmarks/bench_kernels).

# Benchmarking

We provide a master script that setups the entire benchmarking suite for you.

## Setting up Data

To download all the datasets and generate all the indexes needed to run our benchmarking suite, you can use the script [/benchmarks/python_scripts/setup_data.py](/benchmarks/python_scripts/setup_data.py). For this, you need Python 3.11 or higher and install the dependencies in `/benchmarks/python_scripts/requirements.txt`. 

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
cmake . -DPDX_COMPILE_BENCHMARKS
make
```

On the [/benchmarks/CMakeLists.txt](/benchmarks/CMakeLists.txt) file, you can find which `.cpp` files map to which benchmark.


## Complete benchmarking scripts list

### IVF index searches

- PDX IVF ADSampling: `/benchmarks/BenchmarkPDXADSampling`
- PDX IVF ADSampling + SQ8: `/benchmarks/BenchmarkU8PDXADSampling`
- PDX Two-Level IVF ADSampling: `/benchmarks/BenchmarkIVF2ADSampling`
- PDX Two Level IVF ADSampling + SQ8: `/benchmarks/BenchmarkU8IVF2ADSampling`
- FAISS IVF: `/benchmarks/python_scripts/ivf_faiss.py`

All of these programs have two optional parameters:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.
- `<buckets_nprobe>` to specify the `nprobe` parameter on the IVF index, which controls the recall. If not given or `0`, it will use a series of parameters from 2 to 4096 set in the [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

> [!IMPORTANT]   
> Recall that the IVF indexes must be created beforehand by the `setup_data.py` script.

###  Exact Search
- USearch: ```python /benchmarks/python_scripts/exact_usearch.py```
- SKLearn: ```python /benchmarks/python_scripts/exact_sklearn.py```
- FAISS: ```python /benchmarks/python_scripts/exact_faiss.py```

All of these programs have one optional parameter:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

**Notes**: Usearch, SKLearn, and FAISS scripts expect the original `.hdf5` files under the `/downloaded` directory.  Furthermore, they require their respective Python packages (`pip install -r ./benchmarks/python_scripts/requirements.txt`).

## Output
Output is written in a .csv format to the `/benchmarks/results/DEFAULT` directory. Each file contains entries detailing the experiment parameters, such as the dataset, algorithm, kNN, number of queries (`n_queries`), `ivf_nprobe`, and, more importantly, the average runtime per query in ms in the `avg` column. Each benchmarking script will create a file with a different name.

## Kernels Experiment
Visit our playground for PDX vs SIMD kernels [here](./benchmarks/bench_kernels)

## SIGMOD'25
Check the `sigmod` branch.