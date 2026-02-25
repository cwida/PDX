import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch import IndexPDXIVFTreeSQ8

np.random.seed(42)

"""
Filtered Search: only consider a subset of row IDs when searching.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 64
    knn = 100
    selectivity = 0.1  # From 0 to 1
    print(f'Running example: Filtered Search\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}\n- selectivity={selectivity}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXIVFTreeSQ8(num_dimensions=num_dimensions, normalize=True)
    print('Building index...')
    index.build(train)
    print(f'Index built: {index.num_clusters} clusters')
    print(f'Index in-memory size: {index.in_memory_size_bytes / (1024 * 1024):.2f} MB')

    print(f'{len(queries)} queries with PDX (filtered)')
    times = []
    clock = TicToc()
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        # Simulate a predicate: randomly choose a subset of row IDs that "pass"
        passing_row_ids = np.random.choice(
            np.arange(len(train), dtype=np.uint64),
            size=int(len(train) * selectivity),
            replace=False,
        )
        clock.tic()
        index.filtered_search(q, knn, row_ids=passing_row_ids, nprobe=nprobe)
        times.append(clock.toc())
    print(f'PDX med. time at {selectivity} selectivity: {np.median(np.array(times))}')

    # Verify correctness: choose 100 random row IDs and search exhaustively
    passing_row_ids = np.random.choice(
        np.arange(len(train), dtype=np.uint64), size=100, replace=False
    )
    ids, dists = index.filtered_search(
        np.ascontiguousarray(queries[0]), knn, row_ids=passing_row_ids, nprobe=0
    )
    # All returned IDs should be from the passing set
    print('Got correct results?', len(set(passing_row_ids).intersection(set(ids))) == len(ids))
