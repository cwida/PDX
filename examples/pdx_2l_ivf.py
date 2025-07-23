import math
import os
import numpy as np
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXIMI
np.random.seed(42)

"""
PDXearch (pruned search) + ADSampling with a Two-Level IVF index (built with FAISS)
Recall is controled with nprobe parameter
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 24
    knn = 100
    print(f'Running example: PDXearch + ADSampling (Two-Level IVF Flat)\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))
    nbuckets = 4 * math.ceil(math.sqrt(len(train)))

    index = IndexPDXIMI(ndim=num_dimensions, nbuckets=nbuckets, normalize=True)
    print('Preprocessing')
    index.preprocess(train)
    print('Training')
    training_points = nbuckets * 50
    rng = np.random.default_rng()
    training_sample_idxs = rng.choice(len(train), size=training_points, replace=False)
    training_sample_idxs.sort()
    index.train(train[training_sample_idxs])
    print('PDXifying')
    index.add_load(train)
    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn, nprobe=nprobe)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))
    # To check results of first query
    # results = index.search(np.ascontiguousarray(queries[0]), knn, nprobe=nprobe)
    # for result in results:
    #     print(result.index, result.distance)

    print(f'{len(queries)} queries with FAISS')
    times = []
    clock = TicToc()
    results = []
    queries = index.preprocess(queries, inplace=False)
    index.core_index.index.nprobe = nprobe
    for i in range(len(queries)):
        q = np.ascontiguousarray(np.array([queries[i]]))
        clock.tic()
        index.core_index.index.search(q, k=knn)
        times.append(clock.toc())
    print('FAISS med. time:', np.median(np.array(times)))
    # To check results of first query
    # print(index.core_index.index.search(np.array([queries[0]]), k=knn))
