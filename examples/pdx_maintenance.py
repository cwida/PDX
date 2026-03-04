import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch import IndexPDXIVFTreeSQ8

np.random.seed(42)

"""
PDXearch maintenance example: build with 50% of data, then append the remaining 50%.
Uses a two-level IVF index with 8-bit scalar quantization (U8).
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 25
    knn = 20
    print(f'Running example: PDXearch Maintenance (Build 50% + Append 50%)')
    print(f'- D={num_dimensions}, k={knn}, nprobe={nprobe}, dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    n = len(train)
    n_build = n // 2
    n_append = n - n_build

    # Build index with first 50% of data
    index = IndexPDXIVFTreeSQ8(num_dimensions=num_dimensions, normalize=True)
    print(f'\nBuilding index with {n_build}/{n} embeddings...')
    clock = TicToc()
    clock.tic()
    index.build(train[:n_build])
    build_time = clock.toc()
    print(f'Build time: {build_time:.1f} ms')
    print(f'Clusters: {index.num_clusters}')
    print(f'Index size: {index.in_memory_size_bytes / (1024 * 1024):.2f} MB')

    # Append remaining 50% one by one
    print(f'\nAppending {n_append} embeddings...')
    clock.tic()
    for i in range(n_build, n):
        index.append(i, train[i])
    append_time = clock.toc()
    print(f'Append time: {append_time:.1f} ms ({append_time / n_append:.2f} ms/embedding)')
    print(f'Clusters after append: {index.num_clusters}')
    print(f'Index size after append: {index.in_memory_size_bytes / (1024 * 1024):.2f} MB')

    # Search
    print(f'\nSearching {len(queries)} queries...')
    times = []
    clock = TicToc()
    for i in range(len(queries)):
        clock.tic()
        index.search(queries[i], knn, nprobe=nprobe)
        times.append(clock.toc())
    print(f'Median search time: {np.median(np.array(times)):.3f} ms')

    # Show results of first query
    ids, dists = index.search(queries[0], knn, nprobe=nprobe)
    print(f'\nFirst query results (ids):  {ids[:10]}')
    print(f'First query results (dists): {dists[:10]}')
