import math
import os
import numpy as np

from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXADSamplingIVFFlat

"""
Example to store the PDX index and the metadata in a file and how to use it later
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'fashion-mnist-784-euclidean.hdf5'
    num_dimensions = 784
    nprobe = 16
    knn = 10
    print(f'Running example: Persisting PDXADSamplingIVF Index\n- D={num_dimensions}\n- k={knn}\n- nprobe={nprobe}\n- dataset={dataset_name}')

    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    nbuckets = 1 * math.ceil(math.sqrt(len(train)))
    index = IndexPDXADSamplingIVFFlat(ndim=num_dimensions, nbuckets=nbuckets)
    print('Preprocessing')
    index.preprocess(train)
    print('Training')
    index.train(train)
    print('PDXifying and Storing')
    index_path = f'./examples/my_idx.pdx'
    matrix_path = f'./examples/my_idx.matrix'
    index.add_persist(train, index_path, matrix_path)

    print('Restoring')
    del index
    # TODO: Restoring should be a utility and static method that instantiate the appropiate class
    index = IndexPDXADSamplingIVFFlat()
    index.restore(index_path, matrix_path)

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
    results = index.search(queries[0], knn, nprobe=nprobe)
    print(results)

