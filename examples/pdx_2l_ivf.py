import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch import IndexPDXIVFTree

np.random.seed(42)

"""
PDXearch with a two-level IVF index (F32).
Recall is controlled with nprobe parameter.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 25
    knn = 100
    print(f'Running example: PDXearch Two-Level IVF (F32)\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXIVFTree(num_dimensions=num_dimensions, normalize=True)
    print('Building index...')
    index.build(train)
    print(f'Index built: {index.num_clusters} clusters')

    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=nprobe)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))

    # Show results of first query
    ids, dists = index.search(np.ascontiguousarray(queries[0]), knn, nprobe=nprobe)
    print('First query results (ids):', ids[:10])
    print('First query results (dists):', dists[:10])
