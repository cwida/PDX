#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "cppyy",
#   "faiss-cpu",
#   "h5py",
# ]
# ///

from typing import List, Literal
import time
import os
os.environ["EXTRA_CLING_ARGS"] = "-O3 -march=native"

import csv
import cppyy
import cppyy.ll
import numpy as np
import h5py
from faiss import (
    omp_set_num_threads as faiss_set_threads,
    METRIC_L2,
    METRIC_INNER_PRODUCT
)
from faiss.contrib.exhaustive_search import knn as faiss_knn

cppyy.load_library('./kernels.dylib')

cppyy.cppdef("""

template<typename T>
struct KNNCandidate {
    uint32_t index;
    float distance;
};

template<>
struct KNNCandidate<float> {
    uint32_t index;
    float distance;
};

template<>
struct KNNCandidate<uint8_t> {
    uint32_t index;
    uint32_t distance;
};

template<typename T>
struct DistanceType {
    using type = float; // default for f32
};
template<>
struct DistanceType<uint8_t> {
    using type = uint32_t;
};

template<typename T>
struct VectorComparator {
    bool operator() (const KNNCandidate<T>& a, const KNNCandidate<T>& b) {
        return a.distance < b.distance;
    }
};

template<typename T>
struct VectorComparatorInverse {
    bool operator() (const KNNCandidate<T>& a, const KNNCandidate<T>& b) {
        return a.distance > b.distance;
    }
};

enum VectorSearchKernel {
    F32_SIMD_IP,
    F32_SIMD_L2,
    F32_PDX_IP,
    F32_PDX_L2,
    
    U8_SIMD_L2,
    U8_SIMD_IP,
    U8_PDX_L2,
    U8_PDX_IP
};

inline float f32_simd_ip(const float *first_vector, const float *second_vector, const size_t d);
inline float f32_simd_l2(const float *first_vector, const float *second_vector, const size_t d);

inline void f32_pdx_ip(const float *first_vector, const float *second_vector, const size_t d);
inline void f32_pdx_l2(const float *first_vector, const float *second_vector, const size_t d);

inline uint32_t u8_simd_l2(const uint8_t *first_vector, const uint8_t *second_vector, const size_t d);
inline uint32_t u8_simd_ip(const uint8_t *first_vector, const uint8_t *second_vector, const size_t d);

inline void u8_pdx_l2(const uint8_t *first_vector, const uint8_t *second_vector, const size_t d);
inline void u8_pdx_ip(const uint8_t *first_vector, const uint8_t *second_vector, const size_t d);

std::vector<KNNCandidate<float>> standalone_f32(
    const VectorSearchKernel kernel,
    const float *first_vector,
    const float *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn
);

std::vector<KNNCandidate<uint8_t>> standalone_u8(
    const VectorSearchKernel kernel,
    const uint8_t *first_vector,
    const uint8_t *second_vector,
    const size_t d,
    const size_t num_queries,
    const size_t num_vectors,
    const size_t knn
);

""")

def save_results(
        stats,
        metadata,
        results_path,
):
    write_header = True
    if os.path.exists(results_path):
        write_header = False
    f = open(results_path, 'a')
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "kernel", "elapsed_ms",
            "ndim", "n_vectors", "n_queries", "knn"
        ])
    elapsed = float(stats['elapsed_ms'])
    writer.writerow([
        metadata.get("kernel_name", ""), elapsed,
        metadata.get("ndim", 0), metadata.get("n_vectors", 0), metadata.get("query_count", 0), metadata.get("knn", 0)
    ])
    f.close()

def generate_random_vectors(count: int, d: int, dtype = np.float32) -> np.ndarray:
    data = np.array([])
    if dtype == "f32":
        data = np.random.rand(count, d).astype(np.float32)
    elif dtype == "u8":
        data = np.random.randint(0, 128, size=(count, d), dtype=np.uint8)
    return data

def bench_faiss(
        vectors: np.ndarray,
        queries: np.ndarray,
        k: int,
        threads: int,
        query_count: int = 1000,
        warmup_repetitions: int = 5,
) -> dict:
    faiss_set_threads(threads)
    n = vectors.shape[0]

    # Warmup
    for i in range(warmup_repetitions):
        _, matches = faiss_knn(vectors, queries, k, metric=METRIC_INNER_PRODUCT)
    start = time.perf_counter()
    _, matches = faiss_knn(vectors, queries, k, metric=METRIC_INNER_PRODUCT)
    elapsed = time.perf_counter() - start

    computed_distances = n * len(queries)
    recalled_top_match = int((matches[:, 0] == np.arange(n)).sum())
    return {
        "elapsed_ms": elapsed * 1000,
        "computed_distances": computed_distances,
        "visited_members": computed_distances,
        "recalled_top_match": recalled_top_match,
    }

def get_warmup_repetition_n(
        n_vectors: int
) -> int:
    if n_vectors < 256:
        return 1000
    elif n_vectors <= 1024:
        return 100
    elif n_vectors <= 131072:
        return 10
    elif n_vectors <= 1048576:
        return 2
    return 2

def bench_standalone(
        vectors: np.ndarray,
        queries: np.ndarray,
        d: int,
        k: int,
        kernel=cppyy.gbl.VectorSearchKernel.F32_SIMD_IP,
        query_count: int = 1000,
        kernel_name: str = "",
        warmup_repetition: int = 5,
        dtype: str = "f32"
) -> dict:
    start = time.perf_counter()
    # Warmup
    for i in range(warmup_repetition):
        if dtype == "f32":
            result = cppyy.gbl.standalone_f32(
                kernel,
                vectors, queries, d,
                len(queries), len(vectors), k)
        elif dtype == "u8":
            result = cppyy.gbl.standalone_u8(
                kernel,
                vectors, queries, d,
                len(queries), len(vectors), k)
    start = time.perf_counter()
    if dtype == "f32":
        result = cppyy.gbl.standalone_f32(
            kernel,
            vectors, queries, d,
            len(queries), len(vectors), k)
    elif dtype == "u8":
        result = cppyy.gbl.standalone_u8(
            kernel,
            vectors, queries, d,
            len(queries), len(vectors), k)
    elapsed_s = time.perf_counter() - start
    matches = []
    for i in range(0, (len(queries) * k), k):
        knn_candidate = result[i]
        matches.append(knn_candidate.index)
    # Reduce stats
    recalled_top_match = int((matches == np.arange(len(queries))).sum())
    return {
        "elapsed_ms": elapsed_s * 1000,
        "recalled_top_match": recalled_top_match,
    }

def row_major_to_pdx(vectors, block_size=64, dtype="f32") -> np.ndarray:
    V, dims = vectors.shape  # V must be multiple of block_size
    assert(V % 64 == 0)
    chunks = V // block_size
    total_size = V * dims
    result = np.empty(total_size, dtype=vectors.dtype)
    if dtype == "f32":
        cur_offset = 0
        for i in range(chunks):
            chunk = vectors[cur_offset: cur_offset + block_size, :]
            # Flatten chunk in Fortran order and put into result
            result[i * block_size * dims: (i + 1) * block_size * dims] = chunk.flatten(order='F')
            cur_offset += block_size
    elif dtype == "u8":
        assert(dims % 4 == 0)
        cur_offset = 0
        for i in range(chunks):
            chunk = vectors[cur_offset: cur_offset + block_size, :]
            # Flatten chunk in Fortran order and put into result
            result[i * block_size * dims: (i + 1) * block_size * dims] = chunk.reshape(block_size, -1, 4).transpose(1, 0, 2).reshape(-1)
            cur_offset += block_size
    return result

def bench_standalone_pdx(
        vectors: np.ndarray,
        vectors_pdx: np.ndarray,
        queries: np.ndarray,
        d: int,
        k: int,
        kernel,
        query_count: int = 1000,
        kernel_name: str = "",
        warmup_repetition: int = 5,
        dtype: str = "f32"
) -> dict:
    # Warmup
    for i in range(warmup_repetition):
        if dtype == "f32":
            result = cppyy.gbl.standalone_f32(
                kernel,
                vectors_pdx, queries, d,
                len(queries), len(vectors), k)
        elif dtype == "u8":
            result = cppyy.gbl.standalone_u8(
                kernel,
                vectors_pdx, queries, d,
                len(queries), len(vectors), k)
    start = time.perf_counter()
    if dtype == "f32":
        result = cppyy.gbl.standalone_f32(
            kernel,
            vectors_pdx, queries, d,
            len(queries), len(vectors), k)
    elif dtype == "u8":
        result = cppyy.gbl.standalone_u8(
            kernel,
            vectors_pdx, queries, d,
            len(queries), len(vectors), k)
    elapsed_s = time.perf_counter() - start
    matches = []
    for i in range(0, (len(queries) * k), k):
        knn_candidate = result[i]
        matches.append(knn_candidate.index)
    # Reduce stats
    recalled_top_match = int((matches == np.arange(len(queries))).sum())
    return {
        "elapsed_ms": elapsed_s * 1000,
        "recalled_top_match": recalled_top_match,
    }

def read_dataset(
        dataset :str,
        dtype: str,
        benchmark_metadata,
        query_count: int
):
    hdf5_file_name = os.path.join('../datasets/downloaded/', dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    train = np.array(hdf5_file["train"], dtype=np.float32)
    test = np.array(hdf5_file["test"], dtype=np.float32)
    ndims = [len(train[0])]

    tuples_to_take = (len(train) // 64) * 64
    print('Dataset was', len(train), 'and now is', tuples_to_take)
    train = train[:tuples_to_take, :]

    if len(test) < query_count:
        query_count = len(test)
        benchmark_metadata['query_count'] = query_count
    else:
        test = test[:query_count, :]
    benchmark_metadata['count'] = len(train)
    if dtype == "u8": # Global Scalar Quantization with Min-Max
        data_max = max(train.max(), test.max())
        data_min = min(train.min(), test.min())
        data_range = data_max - data_min
        global_scale_factor = 127.0 / data_range
        train = (train - data_min) * global_scale_factor
        test = (test - data_min) * global_scale_factor
        train = train.round(decimals=0).astype(dtype=np.uint8)
        test = test.round(decimals=0).astype(dtype=np.uint8)
    return train, test, len(train), query_count, ndims


def main(
        count: int,
        k: int = 1,
        ndims: List[int] = [256, 1024, 1536],
        threads: int = 1,
        query_count: int = -1,
        output: str = "",
        dtype: str = "f32",
        warmup_repetition: int = -1,
        dataset: str = ""
):
    if query_count > count:
        print('Exiting [query_count > count]')
        return

    benchmark_metadata = {
        "query_count": query_count,
        "n_vectors": count,
        "knn": k,
        "dtype": dtype
    }
    kernels_cpp_simd_f32 = [
        (
            "F32_SIMD_IP",
            cppyy.gbl.f32_simd_ip,
            cppyy.gbl.VectorSearchKernel.F32_SIMD_IP
        ),
        (
            "F32_SIMD_L2",
            cppyy.gbl.f32_simd_l2,
            cppyy.gbl.VectorSearchKernel.F32_SIMD_L2
        )
    ]
    kernels_cpp_simd_u8 = [
        (
            "U8_SIMD_L2",
            cppyy.gbl.u8_simd_l2,
            cppyy.gbl.VectorSearchKernel.U8_SIMD_L2
        ),
        (
            "U8_SIMD_IP",
            cppyy.gbl.u8_simd_ip,
            cppyy.gbl.VectorSearchKernel.U8_SIMD_IP
        )
    ]
    kernels_cpp_pdx_f32 = [
        (
            "F32_PDX_IP",
            cppyy.gbl.f32_pdx_ip,
            cppyy.gbl.VectorSearchKernel.F32_PDX_IP
        ),
        (
            "F32_PDX_L2",
            cppyy.gbl.f32_pdx_l2,
            cppyy.gbl.VectorSearchKernel.F32_PDX_L2
        )
    ]
    kernels_cpp_pdx_u8 = [
        (
            "U8_PDX_L2",
            cppyy.gbl.u8_pdx_l2,
            cppyy.gbl.VectorSearchKernel.U8_PDX_L2
        ),
        (
            "U8_PDX_IP",
            cppyy.gbl.u8_pdx_ip,
            cppyy.gbl.VectorSearchKernel.U8_PDX_IP
        ),
    ]

    kernels_per_dtype_simd = {
        'f32': kernels_cpp_simd_f32,
        'u8': kernels_cpp_simd_u8
    }
    kernels_per_dtype_pdx = {
        'f32': kernels_cpp_pdx_f32,
        'u8': kernels_cpp_pdx_u8
    }

    kernels_cpp_simd = kernels_per_dtype_simd.get(dtype, [])
    kernels_cpp_pdx = kernels_per_dtype_pdx.get(dtype, [])

    if query_count == -1:
        query_count = count

    # Check which dimensions should be covered:
    if dataset != "":
        benchmark_metadata['dataset'] = dataset
        vectors, queries,  count, query_count, ndims = read_dataset(dataset, dtype, benchmark_metadata, query_count)
    for ndim in ndims:
        benchmark_metadata["ndim"] = ndim
        print("-" * 80)
        print(f"Testing {ndim:,}d kernels, {count} vectors, {query_count} queries")
        if dataset == "":
            vectors = generate_random_vectors(count, ndim, dtype)
            queries = vectors[:query_count].copy()

        if warmup_repetition == -1:
            warmup_repetition = get_warmup_repetition_n(len(vectors))

        # Provide FAISS benchmarking baselines:
        # print(f"Profiling FAISS over {count:,} vectors and {query_count} queries with Jaccard metric")
        # benchmark_metadata['kernel_name'] = "FAISS"
        # stats = bench_faiss(
        #     vectors=vectors,
        #     k=k,
        #     threads=threads,
        #     query_count = query_count,
        #     warmup_repetition = warmup_repetition
        # )
        # print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        # Analyze all the kernels:
        for name, _, kernel_id in kernels_cpp_simd:
            # Warmup
            benchmark_metadata['kernel_name'] = name
            print(f"Profiling `{name}` in standalone c++ over {count:,} vectors and {query_count} queries")
            stats = bench_standalone(vectors=vectors, queries=queries, d=ndim, k=k, kernel=kernel_id, query_count=query_count, kernel_name=name, warmup_repetition=warmup_repetition, dtype=dtype)
            if len(output): save_results(stats, benchmark_metadata, output)
            else:
                print(f"- Elapsed: {stats['elapsed_ms']:,.4f} ms")
                if dataset == "": print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        if len(vectors) % 64 == 0:
            vectors_pdx_64 = row_major_to_pdx(vectors, 64, dtype)
            # vectors_pdx_256 = row_major_to_pdx(vectors, 256, dtype)
            for name, _, kernel_id in kernels_cpp_pdx:
                benchmark_metadata['kernel_name'] = name
                print(f"Profiling `{name}` in standalone c++ with the PDX layout over {count:,} vectors and {query_count} queries")
                if "F32" in name:
                    stats = bench_standalone_pdx(vectors=vectors, vectors_pdx=vectors_pdx_64, queries=queries, d=ndim, k=k,
                                                 kernel=kernel_id, query_count=query_count, kernel_name=name,
                                                 warmup_repetition=warmup_repetition, dtype=dtype)
                elif "U8" in name:
                    stats = bench_standalone_pdx(vectors=vectors, vectors_pdx=vectors_pdx_64, queries=queries, d=ndim, k=k,
                                                 kernel=kernel_id, query_count=query_count, kernel_name=name,
                                                 warmup_repetition=warmup_repetition, dtype=dtype)
                if len(output): save_results(stats, benchmark_metadata, output)
                else:
                    print(f"- Elapsed: {stats['elapsed_ms']:,.4f} ms")
                    if dataset == "": print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        description="Comparing HPC kernels for PDX"
    )
    arg_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of vectors to generate for the benchmark",
    )
    arg_parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to search for",
    )
    arg_parser.add_argument(
        "--ndims",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 1536],
        help="List of dimensions to test (e.g., 256, 512, 1024, 1536)",
    )
    arg_parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for the benchmark",
    )
    arg_parser.add_argument(
        "--query_count",
        type=int,
        default=-1,
        help="Number of queries to use for the benchmark",
    )
    arg_parser.add_argument(
        "--output",
        type=str,
        default="",
        help="File path to output the benchmark results",
    )
    arg_parser.add_argument(
        "--dtype",
        type=str,
        default="f32",
        help="Data type to use (f32, u8)",
    )
    arg_parser.add_argument(
        "--warmup",
        type=int,
        default=-1,
        help="Number of search repetitions to warmup the cache",
    )
    arg_parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to use as vectors. If used, overrides query_count to the dataset query count.",
    )
    args = arg_parser.parse_args()
    main(
        count=args.count,
        k=args.k,
        ndims=args.ndims,
        threads=args.threads,
        query_count=args.query_count,
        output=args.output,
        dtype=args.dtype,
        warmup_repetition=args.warmup,
        dataset=args.dataset
    )