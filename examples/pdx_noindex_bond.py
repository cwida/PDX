import numpy as np
import faiss
import os
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXBONDFlat

"""
PDXearch (pruned search) + BOND on the entire collection (no index)
This produces exact results, and does not need to transform the vectors
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'msong-420.hdf5'
    num_dimensions = 420
    knn = 100
    print(f'Running example: PDXearch + BOND (no index)\n- D={num_dimensions}\n- k={knn}\n- dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))
    queries = queries[:100]

    index = IndexPDXBONDFlat(ndim=num_dimensions)
    print('PDXifying Collection')
    index.preprocess(train)
    index.add(train)

    print(f'{len(queries)} queries with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn)
        times.append(clock.toc())
    print('PDX med. time:', np.median(np.array(times)))
    # To check results of first query
    results = index.search(queries[0], knn)
    print(results)

    print(f'{len(queries)} queries with FAISS')
    # We need to normalize the queries
    index.preprocess(queries, inplace=True)
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
    print('FAISS med. time:', np.median(np.array(times)))
    # To check results of first query
    print(faiss_index.search(np.array([queries[0]]), k=knn))
