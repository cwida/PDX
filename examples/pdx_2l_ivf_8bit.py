import math
import os
import numpy as np
import scipy
import faiss
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXIMISQ8
np.random.seed(42)

"""
PDXearch (pruned search) + ADSampling with a Two-Level IVF index (built with FAISS) + 8-bit Scalar Quantization
Recall is controled with nprobe parameter
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'agnews-mxbai-1024-euclidean.hdf5'
    num_dimensions = 1024
    nprobe = 64
    knn = 100
    print(f'Running example: PDXearch + ADSampling (Two-Level IVF SQ8)\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))
    nbuckets = 4 * math.ceil(math.sqrt(len(train)))

    index = IndexPDXIMISQ8(ndim=num_dimensions, nbuckets=nbuckets, normalize=True)
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
        clock.tic()
        index.search(queries[i], knn, nprobe=nprobe)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))
    # To check results of first query
    results = index.search(np.ascontiguousarray(queries[1]), knn, nprobe=nprobe)
    print(results)

    print(f'{len(queries)} queries with FAISS F32')
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
    print(index.core_index.index.search(np.array([queries[1]]), k=knn))

    # Scalar Quantization in FAISS is EXTREMELY slow in ARM due to lack of SIMD
    # print('Training FAISS SQ8')
    # f_index = faiss.IndexScalarQuantizer(num_dimensions, faiss.ScalarQuantizer.QT_8bit)
    # f_index.train(train[training_sample_idxs])
    # f_index.add(train)
    # f_index.nprobe = nprobe
    # print(f'{len(queries)} queries with FAISS U8')
    # for i in range(len(queries)):
    #     q = np.ascontiguousarray(np.array([queries[i]]))
    #     clock.tic()
    #     f_index.search(q, k=knn)
    #     times.append(clock.toc())
    # print('FAISS med. time:', np.median(np.array(times)))
