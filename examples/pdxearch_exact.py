import numpy as np
import faiss
import os
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXADSamplingFlat

"""
PDXearch (pruned search) + ADSampling on the entire collection (no index)
This produces (almost) exact results
The vectors are transformed to improve pruning efficiency
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'msong-420.hdf5'
    num_dimensions = 420
    knn = 10
    print(f'Running example: PDXearch + ADSampling (no index)\n- D={num_dimensions}\n- k={knn}\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXADSamplingFlat(ndim=num_dimensions)

    print('Preprocessing data')
    adsampling_train = index.preprocess(train, inplace=False)
    print('PDXifying')
    index.add_load(adsampling_train)

    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn)
        times.append(clock.toc())
    print('PDX avg. time:', sum(times) / float(len(times)))
    # To check results of first query
    results = index.search(queries[0], knn)
    for result in results:
        print(result.index, result.distance)

    print(f'{len(queries)} queries with FAISS')
    times = []
    clock = TicToc()
    results = []
    faiss_index = faiss.IndexFlatL2(num_dimensions)
    faiss_index.add(train)
    for i in range(len(queries)):
        q = np.ascontiguousarray(np.array([queries[i]]))
        clock.tic()
        faiss_index.search(q, k=knn)
        times.append(clock.toc())
    print('FAISS avg. time:', sum(times) / float(len(times)))
    # To check results of first query
    print(faiss_index.search(np.array([queries[0]]), k=knn))

