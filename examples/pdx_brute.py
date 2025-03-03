import numpy as np
import faiss
import os
from examples_utils import TicToc, read_hdf5_data
from pdxearch.index_factory import IndexPDXFlat

"""
PDX (NO PRUNING) on the entire collection (no index)
The distance calculations are done in the vertical layout. This produces exact results.
Download the .hdf5 data here: https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing
"""
if __name__ == "__main__":
    dataset_name = 'stl-9216.hdf5'
    num_dimensions = 9216
    knn = 10
    print(f'Running example: PDX Brute-force (no index, no pruning)\nD={num_dimensions}k={knn}dataset={dataset_name}')
    train, queries = read_hdf5_data(os.path.join('./benchmarks/datasets/downloaded', dataset_name))

    index = IndexPDXFlat(ndim=num_dimensions)

    print('PDXifying')
    index.add_load(train)

    print('Querying with PDX')
    times = []
    clock = TicToc()
    results = []
    for i in range(len(queries)):
        q = np.ascontiguousarray(queries[i])
        clock.tic()
        index.search(q, knn)
        times.append(clock.toc())
    results = index.search(queries[0], knn)
    for result in results:
        print(result.index, result.distance)
    print('PDX (no pruning) avg. time:', sum(times) / float(len(times)))

    print('Querying with FAISS')
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
    print(faiss_index.search(np.array([queries[0]]), k=knn))

