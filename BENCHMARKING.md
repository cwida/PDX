# Benchmarking

We benchmarked several algorithms in different data layouts using C++. For this, we build and store one index for each benchmarked algorithm. For example, we create one file with the IVF index and the vectors in the PDX layout for ADSampling and another file with the same IVF index but the vectors in the N-ary layout for ADSampling.

Despite this not being space efficient, it lets us re-run benchmarks easily just by reading the proper file. 

Therefore, to set up the data, we:
1. Download the raw files containing the vectors. We use the `.hdf5` format following the convention used in the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks/) project. One `.hdf5` file with two datasets: `train` and `test`.
2. Build a *core* IVF index for each dataset using FAISS.
3. From the *core* index, we read the vectors and apply the respective transformations for (i) ADSampling and (ii) DDC (previously named BSA). For (iii) BOND, we just bypass the raw vectors.
4. We store the preprocessed vectors in a file using one of the following layouts: (i) Dual-block or (ii) PDX. The Dual-block layout is the N-ary layout partitioned in two blocks at Δd (refer to the [ADSampling paper](https://dl.acm.org/doi/pdf/10.1145/3589282) or [our publication](https://ir.cwi.nl/pub/35044/35044.pdf)). For PDX-LINEAR-SCAN (no pruning on PDX), we group vectors in the PDX layout in blocks of 64.

## Master Script

To download all the datasets and generate all the indexes needed to run our benchmarking suite, you can use the script [/benchmarks/python_scripts/setup_data.py](/benchmarks/python_scripts/setup_data.py). For this, you need Python 3.11 or higher and install the dependencies in `/benchmarks/python_scripts/requirements.txt`. 

> [!CAUTION]  
> You will need roughly 300GB of disk for ALL the indexes of the datasets used in our paper.

Run the script from the root folder with the script flags `DOWNLOAD` and `GENERATE_IVF` set to `True` and the values in the `ALGORITHMS` array uncommented. You do not need to generate the `ground_truth` for k=10 as it is already present. You can further uncomment/comment the datasets you wish to create indexes for on the `DATASETS` array [here](/benchmarks/python_scripts/setup_settings.py). 
```sh
pip install -r ./benchmarks/python_scripts/requirements.txt
python ./benchmarks/python_scripts/setup_data.py
```
The indexes will be created under the `/benchmarks/datasets/` directory.

### Downloading the raw vectors `.hdf5` data manually
We also have options if you prefer to manually download the data:
- Download and unzip ALL the 22 `.hdf5` datasets (~25GB zipped and ~40GB unzipped) manually from [here](https://drive.google.com/file/d/1I8pbwGDCSe3KqfIegAllwoP5q6F4ohj2/view?usp=sharing). These include the GloVe variants, arXiv/768, DBPedia/1536, DEEP/96, GIST/960, SIFT/128, MNIST, etc. You must put the unzipped `.hdf5` files inside `/benchmarks/datasets/downloaded`.
- Download datasets individually from [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing). 

Then, run the Master Script with the flag `DOWNLOAD = False`. You can also Uncomment/comment the datasets you wish to create indexes for on the `DATASETS` array [here](/benchmarks/python_scripts/setup_settings.py).


### Random collection of `float32` vectors
For the experiment presented in Section 6.2 of our paper, we generate random collections of vectors. You can generate them by running the Master Script with the flag `GENERATE_SYNTHETIC = True`. Set the other flags to `False` and comment all the values in the `ALGORITHMS` array.  

## Configuring the IVF indexes
Configure the IVF indexes in [/benchmarks/python_scripts/setup_core_index.py](/benchmarks/python_scripts/setup_core_index.py). The benchmarks presented in our publication use `n_buckets = 2 * sqrt(n)` for the number of inverted lists (buckets) and `n_training_points = 50 * n_buckets`. This will create solid indexes fairly quickly. 

> [!NOTE]   
> We have also run experiments with higher `n_buckets` values (`4 * sqrt(n)`, `8 * sqrt(n)`) and training the index with all the points. The effectiveness of the method does not change substantially. 

## Running Benchmarks
Once you have downloaded and created the indexes, you can start benchmarking. 

### Requirements
1. Clang++17 or higher.
2. CMake 3.26 or higher.
3. Set CXX variable. E.g., `export CXX="/usr/bin/clang++-18"`

### Building
In our benchmarks, we build our scripts with the proper `march` flags. You will mostly be fine with `-march=native`. However, for Intel SPR, one has to manually set the vector width to 512 with `-mprefer-vector-width=512`, as LLVM has not yet activated AVX512 by default in this architecture. The latter is due to AVX512 downclocking the CPU on earlier Intel microarchitectures (Ice Lake and earlier). This is not the case anymore. Below are the flags we used for each microarchitecture:
```sh
# GRAVITON4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v2"
# GRAVITON3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v1"
# Intel Sapphire Rapids
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=sapphirerapids -mtune=sapphirerapids -mprefer-vector-width=512"
# ZEN4
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver4 -mtune=znver4"
# ZEN3
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver3 -mtune=znver3"

make
```

All the benchmarking scripts in C++ can be found in the `/benchmarks` directory. We have separated them by algorithm and experiment. On the [/benchmarks/CMakeLists.txt](/benchmarks/CMakeLists.txt) file, you can find which `.cpp` files map to which benchmark. You can build and run these individually. However, we provide shell scripts to make things easier.
- `/benchmarks/benchmark_ivf.sh <python_command>`: `make` and runs benchmarks of all algorithms which run IVF index searches. The only parameter of the script is your `python` command to be able to run FAISS. Make sure to build for the corresponding architecture beforehand. E.g.,
```sh
cmake . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -mcpu=neoverse-v2"
./benchmarks/benchmark_ivf.sh python3.11
```

- `/benchmarks/benchmark_exact.sh <python_command>`: `make` and runs benchmarks of all exact-search algorithms. The only parameter of the script is your `python` command to be able to run FAISS, USearch, Scikit-learn, and PyMilvus.

- `/benchmarks/benchmark_kernels.sh`: `make` and runs benchmarks for the PDX vs SIMD kernels (Section 6.2 of [our paper](https://ir.cwi.nl/pub/35044/35044.pdf)).

## Complete benchmarking scripts list

### IVF index searches

- Scalar Linear Scan (no pruning, no SIMD): `/benchmarks/BenchmarkNaryIVFLinearScan`
- ADSampling (SIMD): `/benchmarks/BenchmarkNaryIVFADSamplingSIMD`
- ADSampling (Scalar): `/benchmarks/BenchmarkNaryIVFADSampling`
- DDC (SIMD): `/benchmarks/BenchmarkNaryIVFBSASIMD`
- PDX ADSampling: `/benchmarks/BenchmarkPDXADSampling`
- PDX DDC: `/benchmarks/BenchmarkPDXBSA`
- PDX BOND: `/benchmarks/BenchmarkPDXIVFBOND`
- FAISS IVF: `/benchmarks/python_scripts/ivf_faiss.py`

All of these programs have two optional parameters:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.
- `<buckets_nprobe>` to specify the `nprobe` parameter on the IVF index, which controls the recall. If not given or `0`, it will use a series of parameters from 2 to 512 set in the [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

PDX BOND has an additional third parameter:
- `<dimension_ordering_criteria>`: An integer value. On Intel SPR, we use distance-to-means (`1`). For the other microarchitectures, we use dimension-zones (`5`). Refer to Figure 5 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf).

> [!IMPORTANT]   
> Recall that the IVF indexes must be created beforehand by the `setup_data.py` script.

###  Exact Search
- PDX BOND: ```/benchmarks/BenchmarkPDXBOND```
- PDX LINEAR SCAN: ```/benchmarks/BenchmarkPDXLinearScan```
- USearch: ```python /benchmarks/python_scripts/exact_usearch.py```
- SKLearn: ```python /benchmarks/python_scripts/exact_sklearn.py```
- Milvus: ```python /benchmarks/python_scripts/exact_milvus.py```
- FAISS: ```python /benchmarks/python_scripts/exact_faiss.py```

All of these programs have one optional parameter:
- `<dataset_name>` to specify the name of the dataset to use. If not given, it will try to use all the datasets set in [benchmark_utils.hpp](/include/utils/benchmark_utils.hpp) or [benchmark_utils.py](/benchmarks/python_scripts/benchmark_utils.py) in the Python scripts.

PDX BOND has an additional second parameter:
- `<dimension_ordering_criteria>`: An integer value. On exact-search, we always use distance-to-means (`1`). Refer to Figure 5 of [our publication](https://ir.cwi.nl/pub/35044/35044.pdf).

**Notes**: Usearch, SKLearn, Milvus, and FAISS scripts expect the original `.hdf5` files under the `/downloaded` directory.  Furthermore, they require their respective Python packages (`pip install -r ./benchmarks/python_scripts/requirements.txt`).

### Kernels Experiment
These kernels DO NOT do a KNN search query. The only work measured is the distance calculation. They also do `WARMUP` runs to warm up the cache.
- PDX+L1: ```/benchmarks/KernelPDXL1```
- PDX+L2: ```/benchmarks/KernelPDXL2```
- PDX+IP: ```/benchmarks/KernelPDXIP```
- SIMD+L1: ```/benchmarks/KernelNaryL1```
- SIMD+L2: ```/benchmarks/KernelNaryL2```
- SIMD+IP: ```/benchmarks/KernelNaryIP```

All these executables have two obligatory parameters:
- `<n_vector>` and `<dimension>`. These determine the random collection to be used for the test. The values are limited to: `n_vectors=(64 128 512 1024 4096 8192 16384 65536 131072 262144 1048576)`,
  `dimensions=(8 16 32 64 128 192 256 384 512 768 1024 1536 2048 4096 8192)`. 

> [!IMPORTANT]   
> Recall that these collections are created by the `setup_data.py` script.

### Milvus IVF
For Milvus IVF, we have to use the standalone version instead of in-memory PyMilvus, as the latter does not support IVF indexes. We have a shell script that installs and runs everything (it also installs Docker and Docker-compose): `/benchmarks/benchmark_milvus.sh`. Note that you need additional storage for this.

### Other experiments
For the other experiments presented in the paper (`GATHER`, `DSM`, `BLOCK_STUDY`, `PHASES`, etc.), check the `sigmod` branch.